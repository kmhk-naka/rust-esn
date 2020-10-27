mod esn;

use esn::ESN;
use gnuplot::{AxesCommon, Color, Figure, Fix};
use ndarray::arr2;
use ndarray::{s, Array1, Array2, Axis};
use std::f64::consts::PI;
use std::fs;

static INIT_LENGTH: usize = 1000;
static TRAIN_LENGTH: usize = 4000;
static TEST_LENGTH: usize = 2000;

fn main() {
    let p = 0.05;
    let input_size = 1;
    let output_size = 1;
    let reservoir_size = 1000;
    let la_ridge = 0.0000001; // 1e-7
    let spr = 0.9;

    let tmax: usize = TRAIN_LENGTH + TEST_LENGTH;

    let mut reservoir = ESN::new(input_size, output_size, reservoir_size, p, spr);

    let data: Array1<f64> = Array1::linspace(0.0, 50.0 * PI, tmax).mapv(f64::sin);

    let mut target: Array1<f64> = Array1::zeros(TRAIN_LENGTH - INIT_LENGTH);
    target.assign(&(data.slice(s![(INIT_LENGTH + 1)..(TRAIN_LENGTH + 1)])));
    let target: Array2<f64> = target.insert_axis(Axis(1));

    let mut train_data: Array1<f64> = Array1::zeros(TRAIN_LENGTH);
    train_data.assign(&(data.slice(s![..TRAIN_LENGTH])));

    let path = String::from("./output");

    if fs::metadata(&path).is_err() {
        match fs::create_dir(&path) {
            Ok(_) => (),
            Err(e) => panic!("Create output directory: {}", e),
        };
    }

    let (val, state) = fit(
        &reservoir_size,
        &output_size,
        &la_ridge,
        &mut reservoir,
        &train_data,
        &target,
        &path,
    );

    predict(&tmax, &output_size, state, val, data, &mut reservoir, &path);
}

fn fit(
    reservoir_size: &usize,
    output_size: &usize,
    la_ridge: &f64,
    reservoir: &mut ESN,
    train_data: &Array1<f64>,
    target: &Array2<f64>,
    path: &String,
) -> (f64, Array2<f64>) {
    let mut X: Array2<f64> = Array2::zeros((TRAIN_LENGTH - INIT_LENGTH, *reservoir_size));
    let mut state: Array2<f64> = Array2::zeros((*reservoir_size, 1));

    for (i, val) in train_data.iter().enumerate() {
        state = reservoir.next_state(&arr2(&[[*val]]), &state);

        if i >= INIT_LENGTH {
            let mut av = X.row_mut(i - INIT_LENGTH);
            av.assign(&state.column(0));
        }
    }
    reservoir.update_w(&X, target, la_ridge);

    let mut outputs: Array2<f64> = Array2::zeros((TRAIN_LENGTH - INIT_LENGTH, *output_size));
    let mut state: Array2<f64> = Array2::zeros((*reservoir_size, 1));
    let mut val: f64 = 0.0;

    for (i, v) in target.iter().enumerate() {
        val = *v;
        state = reservoir.next_state(&arr2(&[[val]]), &state);

        let mut av = outputs.row_mut(i);
        av.assign(&reservoir.output(&state).column(0));
    }

    let mut fg = Figure::new();
    fg.axes2d()
        .lines(INIT_LENGTH..TRAIN_LENGTH, outputs.iter(), &[Color("blue")])
        .lines(INIT_LENGTH..TRAIN_LENGTH, target.iter(), &[Color("red")])
        .set_y_range(Fix(-1.1), Fix(1.1));
    fg.save_to_png(format!("{}/{}", path, "training.png"), 1024, 768)
        .unwrap();

    (val, state)
}

fn predict(
    tmax: &usize,
    output_size: &usize,
    state: Array2<f64>,
    val: f64,
    data: Array1<f64>,
    reservoir: &mut ESN,
    path: &String,
) {
    let mut state: Array2<f64> = state;
    let mut outputs: Array2<f64> = Array2::zeros((TEST_LENGTH, *output_size));

    let mut test_data: Array1<f64> = Array1::zeros(TEST_LENGTH);
    test_data.assign(&(data.slice(s![TRAIN_LENGTH..])));

    let mut y: Array2<f64> = Array2::zeros((*output_size, 1));
    y.fill(val);

    for (i, val) in test_data.iter().enumerate() {
        state = reservoir.next_state(&arr2(&[[*val]]), &state);
        y = reservoir.output(&state);

        let mut av = outputs.row_mut(i);
        av.assign(&y.column(0));
    }

    let mut fg = Figure::new();
    fg.axes2d()
        .lines(TRAIN_LENGTH..*tmax, outputs.iter(), &[Color("blue")])
        .lines(TRAIN_LENGTH..*tmax, test_data.iter(), &[Color("red")])
        .set_y_range(Fix(-1.1), Fix(1.1));
    fg.save_to_png(format!("{}/{}", path, "prediction.png"), 1024, 768)
        .unwrap();
}

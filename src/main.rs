mod esn;

use esn::ESN;
use ndarray::{s, Array1, Array2, Axis};
use std::f64::consts::PI;
use std::fs;

static INIT_LENGTH: usize = 1000;
static TRAIN_LENGTH: usize = 3000;
static TEST_LENGTH: usize = 2000;

fn main() {
    let p = 0.05;
    let input_size = 1;
    let output_size = 1;
    let reservoir_size = 1000;
    let la_ridge = 0.0000001; // 1e-7
    let spr = 0.9;

    let tmax: usize = TRAIN_LENGTH + TEST_LENGTH;

    let path = String::from("./output");

    if fs::metadata(&path).is_err() {
        match fs::create_dir(&path) {
            Ok(_) => (),
            Err(e) => panic!("Create output directory: {}", e),
        };
    };

    // Sine Curve
    let fname = String::from("Sine-Curve");
    let mut reservoir = ESN::new(
        input_size,
        output_size,
        reservoir_size,
        INIT_LENGTH,
        TRAIN_LENGTH,
        TEST_LENGTH,
        p,
        spr,
    );

    let data: Array1<f64> = sine_curve(tmax, 0.0, 50.0 * PI);
    let (target, train_data) = data_proc(&data);
    let (fmin, fmax) = (-1.1, 1.1);

    let state = reservoir.fit(
        &la_ridge,
        &train_data,
        &target,
        &path,
        &fname,
        fmin,
        fmax
    );

    reservoir.predict(
        &tmax,
        state,
        data,
        &path,
        &fname,
        fmin,
        fmax
    );

    // Mackey-Glass System
    let fname = String::from("Mackey-Glass");
    let mut reservoir = ESN::new(
        input_size,
        output_size,
        reservoir_size,
        INIT_LENGTH,
        TRAIN_LENGTH,
        TEST_LENGTH,
        p,
        spr,
    );

    let data: Array1<f64> = mackey_glass(tmax, 0.2, 1.0, 0.9, 17, 10.0, 0.1);
    let (target, train_data) = data_proc(&data);
    let (fmin, fmax) = (0.1, 1.5);

    let state = reservoir.fit(
        &la_ridge,
        &train_data,
        &target,
        &path,
        &fname,
        fmin,
        fmax
    );

    reservoir.predict(
        &tmax,
        state,
        data,
        &path,
        &fname,
        fmin,
        fmax
    );
}

fn data_proc(data: &Array1<f64>) -> (Array2<f64>, Array1<f64>) {
    let mut target: Array1<f64> = Array1::zeros(TRAIN_LENGTH - INIT_LENGTH);
    target.assign(&(data.slice(s![(INIT_LENGTH + 1)..(TRAIN_LENGTH + 1)])));
    let target: Array2<f64> = target.insert_axis(Axis(1));

    let mut train_data: Array1<f64> = Array1::zeros(TRAIN_LENGTH);
    train_data.assign(&(data.slice(s![..TRAIN_LENGTH])));

    (target, train_data)
}

fn sine_curve(n: usize, start: f64, end: f64) -> Array1<f64> {
    Array1::linspace(start, end, n).mapv(f64::sin)
}

fn mackey_glass(n: usize, a: f64, b: f64, c: f64, d: i32, e: f64, initial: f64) -> Array1<f64> {
    let n: usize = n + 300;

    let a: Array1<f64> = Array1::from_elem(n, 0.2);
    let b: Array1<f64> = Array1::from_elem(n, 1.0);
    let c: Array1<f64> = Array1::from_elem(n, 0.9);
    let d: Array1<i32> = Array1::from_elem(n, 20);
    let e: Array1<f64> = Array1::from_elem(n, 10.0);

    let mut x: Array1<f64> = Array1::zeros(n);
    x[0] = initial;

    for k in 0..(n - 1) {
        let idx = if k as i32 - d[k] < 0 {
            n as i32 + k as i32 - d[k]
        } else {
            k as i32 - d[k]
        } as usize;

        let val = c[k] * x[k] + ((a[k] * x[idx]) / (b[k] + (x[idx].powf(e[k]))));
        x[k + 1] = val;
    }

    x.slice(s![300..]).to_owned()
}

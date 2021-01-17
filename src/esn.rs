use gnuplot::{AxesCommon, Caption, Color, Figure, Fix, Font, Graph, Major};
use ndarray::{arr1, arr2, s, Array1, Array2};
use ndarray_linalg as LA;
use ndarray_linalg::*;

pub struct ESN {
    input_size: usize,
    reservoir_size: usize,
    output_size: usize,
    init_length: usize,
    train_length: usize,
    test_length: usize,
    w: Array2<f64>,
    w_in: Array2<f64>,
    w_rec: Array2<f64>,
}

impl ESN {
    pub fn new(
        input_size: usize,
        output_size: usize,
        reservoir_size: usize,
        init_length: usize,
        train_length: usize,
        test_length: usize,
        p: f64,
        spr: f64,
    ) -> ESN {
        let a: Array2<f64> = LA::random((reservoir_size, reservoir_size));
        let a: Array2<f64> = a.map(|x| ((*x <= p) as i32) as f64);
        let w_rec: Array2<f64> = 2.0 * LA::random((reservoir_size, reservoir_size)) - 1.0;
        let w_rec: Array2<f64> = a * w_rec;
        let e: Array1<c64> = w_rec.clone().eigvals().unwrap();
        let max: f64 = e
            .map(|v| v.re())
            .mapv(f64::abs)
            .iter()
            .fold(0.0 / 0.0, |m, v| v.max(m));
        let w_rec: Array2<f64> = w_rec * arr2(&[[spr]]) / arr1(&[max]);

        ESN {
            input_size,
            reservoir_size,
            output_size,
            init_length,
            train_length,
            test_length,
            w_rec: w_rec,
            w_in: LA::random((input_size, reservoir_size)) * 2.0 - 1.0,
            w: LA::random((reservoir_size, output_size)) * 2.0 - 1.0,
        }
    }

    fn next_state(&self, u: &Array2<f64>, x: &Array2<f64>) -> Array2<f64> {
        let next_state: Array2<f64> = (self.w_rec.dot(x) + self.w_in.t().dot(u)).mapv(f64::tanh);
        next_state
    }

    fn output(&self, x: &Array2<f64>) -> Array2<f64> {
        self.w.t().dot(x)
    }

    fn update_w(&mut self, X: &Array2<f64>, target: &Array2<f64>, la_ridge: &f64) {
        let I: Array2<f64> = Array2::eye(self.reservoir_size);
        let inv_ = match (X.t().dot(&*X) + &I * &arr2(&[[*la_ridge]])).inv() {
            Ok(arr) => arr,
            Err(_) => panic!("Cannot invert a Array at RidgeRegression!"),
        };
        let w_out: Array2<f64> = inv_.dot(&X.t()).dot(target);

        self.w = w_out;
    }

    pub fn fit(
        &mut self,
        la_ridge: &f64,
        train_data: &Array1<f64>,
        target: &Array2<f64>,
        path: &String,
        fname: &String,
        fmin: f64,
        fmax: f64,
    ) -> Array2<f64> {
        let mut X: Array2<f64> =
            Array2::zeros((self.train_length - self.init_length, self.reservoir_size));
        let mut state: Array2<f64> = Array2::zeros((self.reservoir_size, 1));

        for (i, val) in train_data.iter().enumerate() {
            state = self.next_state(&arr2(&[[*val]]), &state);

            if i >= self.init_length {
                let mut av = X.row_mut(i - self.init_length);
                av.assign(&state.column(0));
            }
        }
        self.update_w(&X, target, la_ridge);

        let outputs: Array2<f64> = self.output(&X.t().to_owned());

        let mut fg = Figure::new();
        fg.axes2d()
            .lines(
                self.init_length..self.train_length,
                outputs.iter(),
                &[Color("blue"), Caption("output")],
            )
            .lines(
                self.init_length..self.train_length,
                target.iter(),
                &[Color("red"), Caption("target")],
            )
            .set_y_range(Fix(fmin), Fix(fmax));
        fg.save_to_png(format!("{}/{}_training.png", path, fname), 1024, 768)
            .unwrap();

        state
    }

    pub fn predict(
        &mut self,
        tmax: &usize,
        state: Array2<f64>,
        data: Array1<f64>,
        path: &String,
        fname: &String,
        fmin: f64,
        fmax: f64,
    ) {
        let mut state: Array2<f64> = state;
        let mut outputs: Array2<f64> = Array2::zeros((self.test_length, self.output_size));

        let mut test_data: Array1<f64> = Array1::zeros(self.test_length);
        test_data.assign(&(data.slice(s![self.train_length..])));

        let mut y: Array2<f64> = Array2::zeros((self.output_size, 1));

        for (i, val) in test_data.iter().enumerate() {
            state = self.next_state(&arr2(&[[*val]]), &state);
            y = self.output(&state);

            let mut av = outputs.row_mut(i);
            av.assign(&y.column(0));
        }

        let x_step = (*tmax - self.train_length) / 10;
        let x_ticks = (self.train_length as i32..*tmax as i32).step_by(x_step);
        let y_max = (outputs.iter().fold(0.0 / 0.0, |m, v| v.max(m)) * 10.0) as i32;
        let y_min = (outputs.iter().fold(0.0 / 0.0, |m, v| v.max(m)) * 10.0) as i32;
        let y_ticks = (y_min..y_max).step_by(2).map(|x| x as f64 / 10.0);

        let mut fg = Figure::new();
        fg.axes2d()
            .lines(
                self.train_length..*tmax,
                outputs.iter(),
                &[Color("blue"), Caption("output")],
            )
            .lines(
                self.train_length..*tmax,
                test_data.iter(),
                &[Color("red"), Caption("input")],
            )
            .set_y_range(Fix(fmin), Fix(fmax))
            .set_x_ticks_custom(
                x_ticks.map(|x| Major(x as i32, Fix(x.to_string()))),
                &[],
                &[Font("Monospace", 18.)],
            )
            .set_y_ticks_custom(
                y_ticks.map(|x| Major(x as f32, Fix("%.1f".to_string()))),
                &[],
                &[Font("Monospace", 18.)],
            )
            .set_legend(Graph(1.0), Graph(1.0), &[], &[Font("Monospace", 20.)]);
        fg.save_to_png(format!("{}/{}_prediction.png", path, fname), 1280, 720)
            .unwrap();
    }
}

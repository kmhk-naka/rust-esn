use ndarray::{arr1, arr2, Array1, Array2, Axis};
use ndarray_linalg as LA;
use ndarray_linalg::*;

pub struct ESN {
    reservoir_size: usize,
    w: Array2<f64>,
    w_in: Array2<f64>,
    w_rec: Array2<f64>,
}

impl ESN {
    pub fn new(
        input_size: usize,
        output_size: usize,
        reservoir_size: usize,
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
            reservoir_size,
            w_rec: w_rec,
            w_in: LA::random((input_size, reservoir_size)) * 2.0 - 1.0,
            w: LA::random((reservoir_size, output_size)) * 2.0 - 1.0,
        }
    }

    pub fn next_state(&self, u: &Array2<f64>, x: &Array2<f64>) -> Array2<f64> {
        let next_state: Array2<f64> = (self.w_rec.dot(x) + self.w_in.t().dot(u)).mapv(f64::tanh);
        next_state
    }

    pub fn output(&self, x: &Array2<f64>) -> Array2<f64> {
        self.w.t().dot(x)
    }

    pub fn update_w(&mut self, X: &Array2<f64>, target: &Array2<f64>, la_ridge: &f64) {
        let I: Array2<f64> = Array2::eye(self.reservoir_size);
        let inv_ = match (X.t().dot(&*X) + &I * &arr2(&[[*la_ridge]])).inv() {
            Ok(arr) => arr,
            Err(_) => panic!("Cannot invert a Array at RidgeRegression!"),
        };
        let w_out: Array2<f64> = inv_.dot(&X.t()).dot(target);

        self.w = w_out;
    }
}

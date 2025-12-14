use core::f32;
use std::f32::consts::E;
use std::fs::File;
use std::io::Read;
use std::time::{SystemTime, UNIX_EPOCH};

use plotters::chart::ChartBuilder;
use plotters::prelude::{BitMapBackend, IntoDrawingArea, PathElement};
use plotters::series::LineSeries;
use plotters::style::{BLACK, Color, IntoFont, RED, WHITE};

const INPUT_SIZE: usize = 784;
const HIDDEN_SIZE: usize = 128;
const OUTPUT_SIZE: usize = 10;
const LEARNING_RATE: f32 = 0.01;
const EPOCHS: usize = 20;

const TRAIN_IMAGE_PATH: &str = "train-images.idx3-ubyte";
const TRAIN_LABEL_PATH: &str = "train-labels.idx1-ubyte";
const TEST_IMAGE_PATH: &str = "t10k-images.idx3-ubyte";
const TEST_LABEL_PATH: &str = "t10k-labels.idx1-ubyte";

struct MNIST_MLP {
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,
}

impl MNIST_MLP {
    fn new() -> Self {
        let w1 = (0..(HIDDEN_SIZE))
            .map(|_| {
                (0..INPUT_SIZE)
                    .map(|_| rand::random_range(-1.0..=1.))
                    .collect()
            })
            .collect();
        let b1 = vec![0.0; HIDDEN_SIZE];

        let w2 = (0..(OUTPUT_SIZE))
            .map(|_| {
                (0..HIDDEN_SIZE)
                    .map(|_| rand::random_range(-1.0..=1.))
                    .collect()
            })
            .collect();
        let b2 = vec![0.0; OUTPUT_SIZE];

        MNIST_MLP { w1, b1, w2, b2 }
    }

    fn relu(z: &Vec<f32>) -> Vec<f32> {
        z.iter().map(|&x| x.max(0.0)).collect()
    }

    fn d_relu(z: f32) -> f32 {
        if z > 0. { 1. } else { 0. }
    }

    fn sigmoid(z: &Vec<f32>) -> Vec<f32> {
        z.iter().map(|&x| 1. / (1. + E.powf(-x))).collect()
    }

    fn d_sigmoid(z: f32) -> f32 {
        let s = 1. / (1. + (-z).exp());
        s * (1. - s)
    }

    fn softmax(z: &Vec<f32>) -> Vec<f32> {
        let max_z = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = z.iter().map(|&x| E.powf(x - max_z)).collect();
        let sum_exps: f32 = exps.iter().sum();

        exps.iter().map(|&x| x / sum_exps).collect()
    }

    fn dot_product(matrix: &Vec<Vec<f32>>, vector: &Vec<f32>) -> Vec<f32> {
        matrix
            .iter()
            .map(|row| row.iter().zip(vector.iter()).map(|(w, x)| w * x).sum())
            .collect()
    }

    fn vec_add(v1: &Vec<f32>, v2: &Vec<f32>) -> Vec<f32> {
        v1.iter().zip(v2.iter()).map(|(a, b)| a + b).collect()
    }

    fn vec_sub(v1: &Vec<f32>, v2: &Vec<f32>) -> Vec<f32> {
        v1.iter().zip(v2.iter()).map(|(a, b)| a - b).collect()
    }

    fn scalar_mul(v: &Vec<f32>, scalar: f32) -> Vec<f32> {
        v.iter().map(|a| a * scalar).collect()
    }

    fn transpose(matrix: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let rows = matrix.len();
        let cols = matrix[0].len();
        (0..cols)
            .map(|c| (0..rows).map(|r| matrix[r][c]).collect())
            .collect()
    }

    fn one_hot_encode(label: usize) -> Vec<f32> {
        let mut v = vec![0.0; OUTPUT_SIZE];
        if label < OUTPUT_SIZE {
            v[label] = 1.0;
        }
        v
    }

    fn argmax(v: &Vec<f32>) -> usize {
        v.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap_or(0)
    }

    // Raw prediction vector
    fn forward(&self, image: &Vec<f32>) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        // Input to Hidden
        let z1 = Self::vec_add(&Self::dot_product(&self.w1, image), &self.b1);
        let a1 = Self::sigmoid(&z1);

        // Hidden to Output
        let z2 = Self::vec_add(&Self::dot_product(&self.w2, &a1), &self.b2);
        let a2 = Self::softmax(&z2);

        (z1, a1, z2, a2)
    }

    // Integer prediction output
    // Argmaxxed from vector output
    fn predict(&self, image: &Vec<f32>) -> usize {
        let (_, _, _, output_distribution) = self.forward(image);
        Self::argmax(&output_distribution)
    }

    fn calculate_loss(prediction: &Vec<f32>, target: &Vec<f32>) -> f32 {
        prediction
            .iter()
            .zip(target.iter())
            .map(|(a, y)| if *y > 0. { -y * (a + 1e-10).ln() } else { 0.0 })
            .sum()
    }

    // We can store gradients in the MNIST_MLP struct
    // because the fields would be the exact same (just probably renamed)
    #[inline]
    fn backprop(
        &mut self,
        x: &Vec<f32>,
        y: &Vec<f32>,
        z1: &Vec<f32>,
        a1: &Vec<f32>,
        a2: &Vec<f32>,
    ) -> MNIST_MLP {
        // Error at last layer
        // NOTE: Use Cross Entropy Loss
        // NOTE:2 Cross entropy loss is used this is the derivative dummy
        let dz2 = Self::vec_sub(a2, y);

        let mut dw2 = vec![vec![0.0; HIDDEN_SIZE]; OUTPUT_SIZE];
        for r in (0..OUTPUT_SIZE) {
            for c in (0..HIDDEN_SIZE) {
                dw2[r][c] = dz2[r] * a1[c];
            }
        }

        let db2 = dz2.clone();

        let hidden_error = Self::dot_product(&Self::transpose(&self.w2), &dz2);

        let dz1: Vec<f32> = hidden_error
            .iter()
            .zip(z1.iter())
            .map(|(err, z)| err * Self::d_sigmoid(*z))
            .collect();

        let mut dw1 = vec![vec![0.0; INPUT_SIZE]; HIDDEN_SIZE];
        for r in (0..HIDDEN_SIZE) {
            for c in (0..INPUT_SIZE) {
                dw1[r][c] = dz1[r] * x[c];
            }
        }

        let db1 = dz1;

        MNIST_MLP {
            w1: dw1,
            b1: db1,
            w2: dw2,
            b2: db2,
        }
    }

    fn sgd_update(&mut self, gradients: &MNIST_MLP) {
        for r in (0..OUTPUT_SIZE) {
            for c in (0..HIDDEN_SIZE) {
                self.w2[r][c] -= LEARNING_RATE * gradients.w2[r][c]
            }

            self.b2[r] -= LEARNING_RATE * gradients.b2[r];
        }

        for r in (0..HIDDEN_SIZE) {
            for c in (0..INPUT_SIZE) {
                self.w1[r][c] -= LEARNING_RATE * gradients.w1[r][c];
            }

            self.b1[r] -= LEARNING_RATE * gradients.b1[r];
        }
    }

    fn train(&mut self, images: &Vec<Vec<f32>>, labels: &Vec<usize>) -> Vec<f32> {
        println!("Starting training on #{} of samples", images.len());

        let mut loss_history = Vec::new();

        for epoch in 1..=EPOCHS {
            let mut correct_predictions = 0;
            let mut total_epoch_loss = 0.;

            for (image, label) in images.iter().zip(labels.iter()) {
                // image & label preprocessing
                let x = image;
                let y = Self::one_hot_encode(*label);

                // Forward pass
                let (z1, a1, z2, a2) = self.forward(x);

                total_epoch_loss += Self::calculate_loss(&a2, &y);

                if Self::argmax(&a2) == *label {
                    correct_predictions += 1;
                }

                // Backprop:
                let gradients = self.backprop(x, &y, &z1, &a1, &a2);

                self.sgd_update(&gradients);
            }

            let average_loss = total_epoch_loss / images.len() as f32;
            loss_history.push(average_loss);

            let accuracy = correct_predictions as f32 / images.len() as f32 * 100.;
            println!("Epoch #{} complete. Accuracy is {:.2}%", epoch, accuracy);
        }

        loss_history
    }

    fn evaluate(&self, inputs: &Vec<Vec<f32>>, targets: &Vec<usize>) -> (f32, [[usize; 10]; 10]) {
        let mut correct = 0;
        let mut confusion_matrix = [[0; 10]; 10];

        for (x, y) in inputs.iter().zip(targets.iter()) {
            let (_, _, _, final_output) = self.forward(x);

            let predicted_idx = final_output
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;

            let actual_idx = *y;

            if predicted_idx == actual_idx {
                correct += 1;
            }
            confusion_matrix[actual_idx][predicted_idx] += 1;
        }

        let accuracy = correct as f32 / inputs.len() as f32;
        (accuracy, confusion_matrix)
    }
}

fn parse_images(path: String) -> Vec<Vec<f32>> {
    let mut file = File::open(path).expect("Could not find image file");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();

    let magic = u32::from_be_bytes(buffer[0..4].try_into().unwrap());
    let count = u32::from_be_bytes(buffer[4..8].try_into().unwrap()) as usize;
    let rows = u32::from_be_bytes(buffer[8..12].try_into().unwrap()) as usize;
    let cols = u32::from_be_bytes(buffer[12..16].try_into().unwrap()) as usize;

    assert_eq!(
        magic, 2051,
        "Magic number doesn't correspond to ubyte and 3 dimensions"
    );

    let size = rows * cols;
    let mut images = Vec::with_capacity(count);

    for i in 0..count {
        let start = 16 + (i * size);
        let end = start + size;

        let img: Vec<f32> = buffer[start..end]
            .iter()
            .map(|&p| p as f32 / 255.0)
            .collect();

        images.push(img);
    }

    images
}

fn parse_labels(path: String) -> Vec<usize> {
    let mut file = File::open(path).expect("Could not find label file.");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();

    let magic = u32::from_be_bytes(buffer[0..4].try_into().unwrap());
    let count = u32::from_be_bytes(buffer[4..8].try_into().unwrap()) as usize;

    assert_eq!(
        magic, 2049,
        "Magic number doesn't correspond to ubyte and 2 dimensions"
    );

    let mut labels = Vec::with_capacity(count);

    for i in 0..count {
        let label_byte = buffer[8 + i];
        labels.push(label_byte.into());
    }

    labels
}

fn load_mnist(data_folder: String) -> (Vec<Vec<f32>>, Vec<usize>, Vec<Vec<f32>>, Vec<usize>) {
    let train_x = parse_images(data_folder.clone() + TRAIN_IMAGE_PATH);
    let train_y = parse_labels(data_folder.clone() + TRAIN_LABEL_PATH);

    let test_x = parse_images(data_folder.clone() + TEST_IMAGE_PATH);
    let test_y = parse_labels(data_folder.clone() + TEST_LABEL_PATH);

    (train_x, train_y, test_x, test_y)
}

fn plot_loss(loss_history: &Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("training_loss.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_loss = loss_history.iter().cloned().fold(0. / 0., f32::max);
    let min_loss = loss_history.iter().cloned().fold(1. / 0., f32::min);

    let mut chart = ChartBuilder::on(&root)
        .caption("Training Loss", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f32..loss_history.len() as f32, min_loss..max_loss)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            (0..).zip(loss_history.iter()).map(|(x, y)| (x as f32, *y)),
            &RED,
        ))?
        .label("Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

fn main() {
    let mut nn = MNIST_MLP::new();

    // Get data from idx files here
    let path = std::env::args().next_back().expect("Pass a valid path");
    let (train_x, train_y, test_x, test_y) = load_mnist(path);
    println!("Successfully loaded dataset");

    // Cross validate / create train and test here
    // NOTE: Since initial sample is already quite large, we'll skip this for now

    println!("Starting training");
    let loss_history = nn.train(&train_x, &train_y);

    // Validate on test set and plot results using plotters including the test and train accuracy,
    // loss, confusion matrix and so on
    // I think currently we only have enough info to print accuracy let me know
    println!("Validating with test set");
    let (accuracy, confusion) = nn.evaluate(&test_x, &test_y);

    println!("Final test accuracy: {:.2}%", accuracy * 100.);
    println!("Confusion matrix (actual \\ predicted)");
    println!("   0  1  2  3  4  5  6  7  8  9");
    for (i, row) in confusion.iter().enumerate() {
        print!("{}: ", i);
        for val in row {
            print!("{:2} ", val); // Simple formatting
        }
        println!();
    }

    if let Err(e) = plot_loss(&loss_history) {
        eprintln!("Error generating plot: {}", e);
    }
}

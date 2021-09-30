use std::{
    fmt::{Display, Formatter},
    fs::File,
    io::{Cursor, Read},
};

use byteorder::{BigEndian, ReadBytesExt};
use elisabeth::u4;
use flate2::read::GzDecoder;
use ndarray::Array2;

#[derive(Debug)]
struct FmnistData {
    sizes: Vec<i32>,
    data: Vec<u8>,
}

impl FmnistData {
    fn new(f: &File) -> Result<FmnistData, std::io::Error> {
        let mut gz = GzDecoder::new(f);
        let mut contents: Vec<u8> = Vec::new();
        gz.read_to_end(&mut contents)?;
        let mut r = Cursor::new(&contents);

        let magic_number = r.read_i32::<BigEndian>()?;

        let mut sizes: Vec<i32> = Vec::new();
        let mut data: Vec<u8> = Vec::new();

        match magic_number {
            2049 => {
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            2051 => {
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            _ => panic!(),
        }

        r.read_to_end(&mut data)?;

        Ok(FmnistData { sizes, data })
    }
}

#[derive(Debug)]
pub struct MnistImage {
    pub pixels: Array2<u4>,
    pub classification: u8,
}

impl MnistImage {
    pub fn label(&self) -> String {
        (match self.classification {
            0 => "T-shirt/top",
            1 => "Trouser",
            2 => "Pullover",
            3 => "Dress",
            4 => "Coat",
            5 => "Sandal",
            6 => "Shirt",
            7 => "Sneaker",
            8 => "Bag",
            9 => "Ankle Boot",
            _ => "Unknown",
        })
        .to_string()
    }
}

impl Display for MnistImage {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        let mut s = self.label();
        s.push_str(": \n");
        for row in self.pixels.rows().into_iter() {
            for c in row {
                s.push_str(&format!("{} ", c.0));
            }
            s.push_str("\n");
        }
        write!(f, "{}", s)
    }
}

pub fn load_data(
    dataset_name: &str,
    quantization: Option<u8>,
) -> Result<Vec<MnistImage>, std::io::Error> {
    let filename = format!("data/{}-labels-idx1-ubyte.gz", dataset_name);
    let label_data = &FmnistData::new(&(File::open(filename))?)?;
    let filename = format!("data/{}-images-idx3-ubyte.gz", dataset_name);
    let images_data = &FmnistData::new(&(File::open(filename))?)?;

    let mut images = Vec::new();
    let image_size = (images_data.sizes[1] * images_data.sizes[2]) as usize;

    for i in 0..images_data.sizes[0] as usize {
        let start = i * image_size;
        let image_data = images_data.data[start..start + image_size]
            .into_iter()
            .map(|x| {
                if let Some(n_bits) = quantization {
                    u4((*x as f64 / 255. * (2_f64.powi(n_bits as i32) - 1.)).round() as u8)
                } else {
                    u4(*x)
                }
            })
            .collect();
        images.push(
            Array2::from_shape_vec(
                (images_data.sizes[1] as usize, images_data.sizes[1] as usize),
                image_data,
            )
            .unwrap(),
        );
    }

    let classifications: Vec<u8> = label_data.data.clone();

    let mut ret: Vec<MnistImage> = Vec::new();

    for (image, classification) in images.into_iter().zip(classifications.into_iter()) {
        ret.push(MnistImage {
            pixels: image,
            classification,
        })
    }

    Ok(ret)
}

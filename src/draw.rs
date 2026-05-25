use image::{DynamicImage, Pixel, Rgba, RgbaImage};

use crate::types::Detections;

pub trait DrawBoxesExt {
    fn draw_boxes(&mut self, detections: &Detections);
}

impl DrawBoxesExt for DynamicImage {
    fn draw_boxes(&mut self, detections: &Detections) {
        let mut rgba = self.to_rgba8();
        rgba.draw_boxes(detections);
        *self = DynamicImage::ImageRgba8(rgba);
    }
}

impl DrawBoxesExt for RgbaImage {
    fn draw_boxes(&mut self, detections: &Detections) {
        for detection in detections {
            let color = color_for_class(detection.class_id);
            draw_rect(self, detection.bbox, color);
            draw_label_chip(self, detection.bbox, color);
        }
    }
}

fn color_for_class(class_id: usize) -> Rgba<u8> {
    const COLORS: [Rgba<u8>; 10] = [
        Rgba([255, 56, 56, 255]),
        Rgba([255, 157, 151, 255]),
        Rgba([255, 112, 31, 255]),
        Rgba([255, 178, 29, 255]),
        Rgba([207, 210, 49, 255]),
        Rgba([72, 249, 10, 255]),
        Rgba([146, 204, 23, 255]),
        Rgba([61, 219, 134, 255]),
        Rgba([26, 147, 52, 255]),
        Rgba([0, 212, 187, 255]),
    ];
    COLORS[class_id % COLORS.len()]
}

fn draw_rect(image: &mut RgbaImage, bbox: crate::types::BoundingBox, color: Rgba<u8>) {
    let width = image.width();
    let height = image.height();
    if width == 0 || height == 0 {
        return;
    }

    let x0 = bbox.x.floor().max(0.0) as u32;
    let y0 = bbox.y.floor().max(0.0) as u32;
    let x1 = bbox.x2().ceil().min(width.saturating_sub(1) as f32) as u32;
    let y1 = bbox.y2().ceil().min(height.saturating_sub(1) as f32) as u32;
    if x0 > x1 || y0 > y1 {
        return;
    }

    for thickness in 0..2 {
        let top = (y0 + thickness).min(y1);
        let bottom = y1.saturating_sub(thickness);
        for x in x0..=x1 {
            image.put_pixel(x, top, color);
            image.put_pixel(x, bottom, color);
        }

        let left = (x0 + thickness).min(x1);
        let right = x1.saturating_sub(thickness);
        for y in y0..=y1 {
            image.put_pixel(left, y, color);
            image.put_pixel(right, y, color);
        }
    }
}

fn draw_label_chip(image: &mut RgbaImage, bbox: crate::types::BoundingBox, color: Rgba<u8>) {
    let x0 = bbox.x.floor().max(0.0) as u32;
    let y0 = bbox.y.floor().max(0.0) as u32;
    let chip_width = bbox.width.clamp(18.0, 72.0) as u32;
    let chip_height = 8;
    let y = y0.saturating_sub(chip_height);

    for yy in y..(y + chip_height).min(image.height()) {
        for xx in x0..(x0 + chip_width).min(image.width()) {
            let mut pixel = image.get_pixel(xx, yy).to_rgba();
            pixel.blend(&Rgba([color[0], color[1], color[2], 210]));
            image.put_pixel(xx, yy, pixel);
        }
    }
}

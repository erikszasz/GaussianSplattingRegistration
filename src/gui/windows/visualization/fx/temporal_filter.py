from PySide6.QtGui import QPixmap, QColor


class TemporalFilter:
    def __init__(self, decay=0.95, max_frames=5):
        self.decay = decay  # Decay factor for old pixels
        self.max_frames = max_frames  # Max frames to store in the buffer
        self.accumulated_color = None  # Accumulated color values
        self.frame_count = 0  # Number of frames processed

    def apply_temporal_filter(self, new_frame: QPixmap):
        if self.frame_count % 30 == 0:
            self.accumulated_color = None

        img = new_frame.toImage()
        if self.accumulated_color is None:
            self.accumulated_color = img.copy()
            return new_frame

        output_image = img.copy()

        for x in range(img.width()):
            for y in range(img.height()):
                current_color = QColor(img.pixel(x, y))
                accumulated_color = QColor(self.accumulated_color.pixel(x, y))

                new_r = int(current_color.red() * (1 - self.decay) + accumulated_color.red() * self.decay)
                new_g = int(current_color.green() * (1 - self.decay) + accumulated_color.green() * self.decay)
                new_b = int(current_color.blue() * (1 - self.decay) + accumulated_color.blue() * self.decay)

                output_image.setPixel(x, y, QColor(new_r, new_g, new_b).rgba())

        self.accumulated_color = output_image
        self.frame_count += 1

        return QPixmap.fromImage(output_image)

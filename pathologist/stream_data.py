from tensorflow.keras.preprocessing.image import ImageDataGenerator


def make_train_data_generator(
    *,
    rotation_range: int = 45,
    horizontal_flip: bool = True,
    vertical_flip: bool = True,
    width_shift_range: float = 0.2,
    height_shift_range: float = 0.2,
    zoom_range: float = 0.1,
    shear_range: float = 0.1,
) -> tuple:
    return ImageDataGenerator(
        rotation_range=rotation_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        zoom_range=zoom_range,
        shear_range=shear_range,
    )

class CropMeta():
    def __init__(self, crop_size, crop_center):
        self.width = crop_size[1]
        self.height = crop_size[0]
        self.center_x = crop_center[1]
        self.center_y = crop_center[0]

    def __str__(self):
        return f'{self.width};{self.height};{self.center_x};{self.center_y}'


def convert_str_to_crop_meta(str_crop_meta):
    parts = str_crop_meta.split(';')
    crop_size = (int(parts[1]), int(parts[0]))
    crop_center = (int(parts[3]), int(parts[2]))
    return CropMeta(crop_size, crop_center)

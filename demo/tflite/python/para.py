models = ["yolox_nano", "yolox_tiny", "yolox_s",
          "yolox_x", "yolox_m", "yolox_l"]


class Para:
    """Base parameters with different yolox model"""
    def __init__(self, model):
        self.num_classes = 80
        self.act = "silu"
        self.input_size = (640, 640)
        self.in_channels = [256, 512, 1024]
        self.depthwise = False

        if model == models[0]:
            self.depth = 0.33
            self.width = 0.25
            self.depthwise = True

        elif model == models[1]:
            self.depth = 0.33
            self.width = 0.375

        elif model == models[2]:
            self.depth = 0.33
            self.width = 0.50

        elif model == models[3]:
            self.depth = 1.33
            self.width = 1.25

        elif model == models[4]:
            self.depth = 0.67
            self.width = 0.75

        elif model == models[5]:
            self.depth = 1.0
            self.width = 1.0

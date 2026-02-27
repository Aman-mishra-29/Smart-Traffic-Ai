from app.camera import capture_frame
from app.model import predict_density
from app.controller import control_traffic

while True:
    img_path = capture_frame()  # capture frame.jpg
    density = predict_density(img_path)  # Light / Moderate / Heavy
    control_traffic(density)  # Adjust GPIO or simulate
    
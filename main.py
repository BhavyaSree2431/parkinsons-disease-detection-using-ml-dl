import numpy as np
from src.fusion_model import fusion_prediction


def main():

    print("\nParkinson Disease Detection System")
    print("----------------------------------")

    # test image path
    image_path = "test_image.png"

    # voice feature vector (753 features)
    voice_features = np.random.rand(753)

    # run fusion model
    fusion_prediction(image_path, voice_features)


if __name__ == "__main__":
    main()
import torch
import matplotlib.pyplot as plt

from PIL import Image
from utils.utils import tensor_to_image
from facenet_pytorch import MTCNN, InceptionResnetV1


class FaceRecognitionSystem:
    def __init__(self):
        self.mtcnn = MTCNN(image_size=160, keep_all=True)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

    def get_embedding_and_face(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image from {image_path}: {e}")
            return None, None

        faces, probs = self.mtcnn(image, return_prob=True)
        if faces is None or len(faces) == 0:
            return None, None

        embedding = self.resnet(faces[0].unsqueeze(0))
        return embedding, faces[0]

    def find_most_similar(self, target_image_path, candidate_image_paths):
        target_emb, target_face = self.get_embedding_and_face(target_image_path)
        if target_emb is None:
            raise ValueError("No face detected in the target image.")

        highest_similarity = float('-inf')
        most_similar_face = None
        most_similar_image_path = None

        candidate_faces = []
        similarities = []

        for candidate_image_path in candidate_image_paths:
            candidate_emb, candidate_face = self.get_embedding_and_face(candidate_image_path)
            if candidate_emb is None:
                similarities.append(None)
                candidate_faces.append(None)
                continue

            similarity = torch.nn.functional.cosine_similarity(target_emb, candidate_emb).item()
            similarities.append(similarity)
            candidate_faces.append(candidate_face)

            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_face = candidate_face
                most_similar_image_path = candidate_image_path

        plt.figure(figsize=(12, 8))

        plt.subplot(2, len(candidate_image_paths) + 1, 1)
        plt.imshow(tensor_to_image(target_face))
        plt.title("Target Image")
        plt.axis("off")

        if most_similar_face is not None:
            plt.subplot(2, len(candidate_image_paths) + 1, 2)
            plt.imshow(tensor_to_image(most_similar_face))
            plt.title("Most Similar")
            plt.axis("off")

        for idx, (candidate_face, similarity) in enumerate(zip(candidate_faces, similarities)):
            plt.subplot(2, len(candidate_image_paths) + 1, idx + len(candidate_image_paths) + 2)
            if candidate_face is not None:
                plt.imshow(tensor_to_image(candidate_face))
                plt.title(f"Score: {similarity * 100:.2f}%")
            else:
                plt.title("No Face")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig('results/result.png')

        if most_similar_image_path is None:
            raise ValueError("No faces detected in the candidate images.")

        return most_similar_image_path, highest_similarity
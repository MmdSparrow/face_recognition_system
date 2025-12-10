from face_recognition_system import FaceRecognitionSystem


image_url_target = 'dataset/target.jpg'
candidate_image_urls = [
    'dataset/negative-1.jpg',
    'dataset/negative-2.jpg',
    'dataset/negative-3.jpg',
    'dataset/positive-1.jpg',
]

face_recognition_system = FaceRecognitionSystem()

most_similar_image, similarity_score = face_recognition_system.find_most_similar(image_url_target, candidate_image_urls)
print(f"The most similar image is: {most_similar_image}")
print(f"Similarity score: {similarity_score * 100:.2f}%")

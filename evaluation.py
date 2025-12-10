image_url_target = 'https://d1mnxluw9mpf9w.cloudfront.net/media/7588/4x3/1200.jpg'
candidate_image_urls = [
    'https://beyondthesinglestory.wordpress.com/wp-content/uploads/2021/04/elon_musk_royal_society_crop1.jpg',
    'https://cdn.britannica.com/56/199056-050-CCC44482/Jeff-Bezos-2017.jpg',
    'https://cdn.britannica.com/45/188745-050-7B822E21/Richard-Branson-2003.jpg'
]

most_similar_image, similarity_score = find_most_similar(image_url_target, candidate_image_urls)
print(f"The most similar image is: {most_similar_image}")
print(f"Similarity score: {similarity_score * 100:.2f}%")

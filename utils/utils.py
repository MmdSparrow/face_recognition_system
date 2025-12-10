def tensor_to_image(tensor):
    image = tensor.permute(1, 2, 0).detach().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype('uint8')
    return image
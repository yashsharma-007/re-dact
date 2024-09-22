from PIL import Image

def remove_metadata(input_path, output_path):
    image = Image.open(input_path)
    
    
    data = list(image.getdata())
    clean_image = Image.new(image.mode, image.size)
    clean_image.putdata(data)
    
    clean_image.save(output_path)
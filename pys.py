

url = 'https://media.istockphoto.com/vectors/illustration-of-maze-labrinth-isolated-on-white-background-medium-vector-id881659914?k=20&m=881659914&s=612x612&w=0&h=YZJ9RmoUyQpQwkpBLW_TgGBJa-XbMhLBNJfm7CVdpkM='

import requests
import mimetypes

response = requests.get(url)
content_type = response.headers['content-type']
extension = mimetypes.guess_extension(content_type)
print(extension)
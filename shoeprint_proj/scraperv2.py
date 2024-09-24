# FOR SCRAPING IMAGES 
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import requests
import io
from PIL import Image
import time

PATH = './final_dataset/'
test_path = "C:\\Users\\gichu\\Documents\\shoeprint_proj\\test_destination"
wd = webdriver.Chrome()



# all shoe urls to be scraped
outsole_urls = {

	"jordan_14" :"https://www.google.com/search?q=bottom+of+jordan+14s&sca_esv=354005c37d90d315&sca_upv=1&udm=2&biw=1536&bih=791&sxsrf=ACQVn0-X861a6RtzwVl5kzdNx342zZ0VhQ%3A1712864673428&ei=oT0YZpK2GY_ui-gP_KCLqA8&ved=0ahUKEwjSiqGT9rqFAxUP9wIHHXzQAvUQ4dUDCBA&uact=5&oq=bottom+of+jordan+14s&gs_lp=Egxnd3Mtd2l6LXNlcnAiFGJvdHRvbSBvZiBqb3JkYW4gMTRzSOEOUPYKWNANcAF4AJABAJgBTqABjAGqAQEyuAEDyAEA-AEBmAIBoAJAwgIFEAAYgASYAwCIBgGSBwExoAeBAQ&sclient=gws-wiz-serp",
	"Jordan_1" :"https://www.google.com/search?q=jordan+1+outsole&tbm=isch&ved=2ahUKEwjG5pSN3pGEAxWRTaQEHRSbBFIQ2-cCegQIABAA&oq=jordan+1+outsole&gs_lp=EgNpbWciEGpvcmRhbiAxIG91dHNvbGUyBRAAGIAEMgUQABiABDIGEAAYBxgeMgYQABgHGB4yBhAAGAcYHjIGEAAYBxgeMgYQABgHGB4yBhAAGAcYHjIGEAAYBRgeMgcQABiABBgYSNcOULgIWMsJcAB4AJABAJgBxAGgAeICqgEDMi4xuAEDyAEA-AEBigILZ3dzLXdpei1pbWfCAgQQIxgniAYB&sclient=img&ei=nom_ZcafGJGbkdUPlLaSkAU&bih=791&biw=1536",

}

def scroll(delay):
	wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
	time.sleep(delay)
    
def get_images(brand, url, num_images, delay):
	image_urls =set() # inorder to keep track of the images we have visited 
	temp = set()

	wd.get(url) # get page

	try:
		accept_button = wd.find_element(By.ID, "L2AGLb") # accepting googles pop up window
		accept_button.click()

	except:
		print("no popup window")

	time.sleep(delay) # wait for page to load
	scroll(delay) # double scrolling to avoid the same images again

	while len(temp) < (num_images): # whilst we still want to collect more images
		scroll(delay)

		thumbnails = wd.find_elements(By.CLASS_NAME, "YQ4gaf") # caution as classname can change and break the program

		for img in thumbnails:

			if len(temp) == (num_images):
				break	

			try:
				img.click()
				time.sleep(delay)
			except:
				print("Failed to click image")

			#time.sleep()
			
			# now that we click the image and select the one with a src tag as it will link to the image
			images = wd.find_elements(By.CSS_SELECTOR, ".sFlh5c.pT0Scc.iPVvYb") 
			for image in images:

				# if we already have the image url then movee on to the next
				if image.get_attribute('src') in image_urls:
					break

				if image.get_attribute('src') and 'http' in image.get_attribute('src'):
					temp.add((brand, image.get_attribute('src')))
					print(f"Image added, {len(temp)} items in dataset")
					

		image_urls = image_urls.union(temp) # add additional image urls to the list containing all image urls

	return image_urls

# This function takes all the image urls we have scraped and downloads them to the destination folder		
def download(image_url, destination, filename):
	try:
		image_content = requests.get(image_url).content
		image_file = io.BytesIO(image_content)
		image = Image.open(image_file)
		final_file_path = destination + filename
		print(final_file_path)

		if image.mode == 'RGBA':
			image = image.convert('RGB')

		with open(final_file_path, "wb") as f:
			image.save(f, "JPEG")

		print("Downloaded :)")

	except Exception as e:
		print('Failed :(', e)


for brand, url in outsole_urls.items():
	counter = 0
	images_to_download = get_images(brand, url, 3, 2)

	for image_src in images_to_download:
		print(image_src)
		download(image_src[1], test_path, f"{brand}_{counter}.jpg" )
		counter += 1



wd.quit()
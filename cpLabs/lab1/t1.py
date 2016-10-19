import cv2
from cpLabs.lab1.demosaic import linear_demosaic, slow_linear_demosaic, slow_edge_based_demosaic
from cpLabs.lab1.gamma_correction import gamma_correction

def main():
	while True:
		selection = __image_menu()
		if selection:
			repeat = __action_menu(selection[0], selection[1])
			if not repeat:
				return
		else:
			return

def __action_menu(img, pattern_shift):
	img = cv2.imread('cpLabs/static/images/lighthouse_RAW_noisy_sigma0.01.png')
	img_cp = img.copy()

	options = [
		['slow_linear_demosaic'],
		['linear_demosaic'],
		['slow_edge_based_demosaic'],
		['gamma_correction'],
		['Save image'],
		['Reset image'],
		['Change image'],
		['Exit']]

	while True: 
		print 'Choose action'
		for i in range(0, len(options)): 
			print str(i+1)+'.', options[i][0]

		input_string = raw_input('>>> ') 
		try:
			option = int(input_string)-1
		except ValueError:
			option = -1

		if option in range(0,len(options)):
			print '>', options[option][0]

		if option==0:
			slow_linear_demosaic(img)
			__show_image(img)
		elif option==1:
			linear_demosaic(img)
			__show_image(img)
		elif option==2:
			slow_edge_based_demosaic(img)
			__show_image(img)
		elif option==3:
			gamma = __input_gamma()
			gamma_correction(img, gamma)
			__show_image(img)
		elif option==len(options)-4: 
			name = __input_image_name()
			cv2.imwrite('output/'+name+'.png',img)
		elif option==len(options)-3: 
			img=img_cp.copy()
		elif option==len(options)-2: 
			return True
		elif option==len(options)-1: 
			return False
		else: 
			print '> Unknown option selected!'
			continue

def __input_gamma():
	while True: 
		print 'Input gamma (0.0 < x < 100.0))'
		input_string = raw_input('>>> ') 
		try:
			option = float(input_string)
		except ValueError:
			option = None
		
		if option:
			return option
		else:
			print 'Wrong value!'

import re
def __input_image_name():
	while True:
		print 'Input filename'
		input_string = raw_input('>>> ')
		if re.match("[A-Za-z0-9-_]+$", input_string):
			return input_string
		else:
			print 'Illegal filename! Should match ^[A-Za-z0-9-_]+$'

def __image_menu():
	options = [
		['lighthouse_RAW_noisy_sigma0.01.png', 'cpLabs/static/images/lighthouse_RAW_noisy_sigma0.01.png', [1,0]],
		['signs.png', 'cpLabs/static/images/raw/signs.png', [0,1]],
		['signs-small.png', 'cpLabs/static/images/raw/signs-small.png', [0,1]],
		['text.png', 'cpLabs/static/images/raw/text.png', [0,1]],
		['text2.png', 'cpLabs/static/images/raw/text2.png', [0,1]],
		['Exit']]

	while True: 
		print 'Choose image'
		for i in range(0, len(options)): 
			print str(i+1)+'.', options[i][0]

		input_string = raw_input('>>> ') 
		try:
			option = int(input_string)-1
		except ValueError:
			option = -1

		if option in range(0, len(options)-1):
			print '>', options[option][0]
			return options[option][1:]
		elif option==len(options)-1: 
			print '> Exit'
			return None
		else: 
			print 'Unknown option selected!'
			continue

def __show_image(img):

	cv2.imshow('Image',img)
	cv2.waitKey(0)
	# HACK
	# don't know why one call isn't work
	for i in range(1,10):
	    cv2.destroyAllWindows()
	    cv2.waitKey(1)

main()
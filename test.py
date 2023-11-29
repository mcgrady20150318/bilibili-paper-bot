import os
path = os.getcwd() + '/'
print(path)
os.system('pdflatex ' + path + 'test.tex')

#specify the base image
FROM python:3.11

#add files to the image
ADD requirements.txt .
ADD __main__.py .

#install the dependencies
RUN pip install -r requirements.txt

#specify the entry command
#python = executable
#main.py = script to run
CMD ["python", "__main__.py"]
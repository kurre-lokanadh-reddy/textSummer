



from __future__ import unicode_literals
from flask import Flask,render_template,url_for,request


import time

# Web Scraping Pkg
from bs4 import BeautifulSoup
# from urllib.request import urlopen
from urllib.request import urlopen

# main model implementaion file 
import text_summerizer as TS

# Reading Time
def readingTime(mytext):
	#total_words = len([ token.text for token in nlp(mytext)])
	total_words = sum([len(sent.split(" ")) for sent in mytext.split(".")])
	estimatedTime = total_words/200.0
	return estimatedTime

# Fetch Text From Url
def get_text(url):
	page = urlopen(url)
	soup = BeautifulSoup(page)
	fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
	return fetched_text

app = Flask(__name__)
@app.route("/")
def tohome():
	return render_template("base.html",contentFile="home.html")
@app.route("/home")
def tohome2():
	return render_template("base.html",contentFile="home.html")
@app.route("/documents")
def docs():
	return render_template("base.html",contentFile="docs.html")

@app.route("/examples")
def exmp():
	return render_template("base.html",contentFile="exmp.html")

@app.route("/aboutus")
def aboutus():
	return render_template("base.html",contentFile="aboutus.html")

'''
@app.route("/modelOne",methods=["GET","POST"])
def modelOne():
	start = time.time()
	rawtext = "EXAMPLE !!!--->"
	final_summary = "EXAMPLE !!!--->"
	final_time=1.00
	final_reading_time=1.30
	summary_reading_time=0.24
	if request.method == 'POST':
		rawtext = request.form.get('raw_text')
		if rawtext== None:
			rawtext = get_text(request.form.get('url_link'))
		if rawtext==None or rawtext=="":
			rawtext="sorry not text detected"
			final_summary="sorry not text detected"
		final_reading_time = readingTime(rawtext)
		if final_summary!="sorry not text detected":
			final_summary=TS.summerize("TextRank",rawtext)
		summary_reading_time = readingTime(final_summary)
		end = time.time()
		final_time = end-start
	return render_template('base.html',contentFile="modelOne.html",otext=rawtext,stext=final_summary,otime=final_reading_time,stime=summary_reading_time)

'''


@app.route("/modelOne",methods=["GET","POST"])
def modelOne():
	start = time.time()
	rawtext = "EXAMPLE !!!--->"
	final_summary = "EXAMPLE !!!--->"
	final_time=1.00
	final_reading_time=1.30
	summary_reading_time=0.14
	if request.method == 'POST':
		modelSelected = request.form.get('model')
		rawtext = request.form['raw_text']
		raw_url = request.form['url_link']
		f=open("logs.txt",'a',encoding="utf-8")
		if rawtext== None or rawtext=="":
			rawtext = get_text(raw_url)
		f.write("text==="+rawtext)
		final_reading_time = readingTime(rawtext)
		final_summary=TS.summerize(modelSelected , rawtext)
		summary_reading_time = readingTime(final_summary)
		f.write("\nsummery==="+final_summary+"\n\n\n")
		end = time.time()
		final_time = end-start
	return render_template('base.html',contentFile="modelOne.html",otext=rawtext,stext=final_summary,otime=final_reading_time,stime=summary_reading_time)




if __name__=="__main__":
    app.run(debug=True)
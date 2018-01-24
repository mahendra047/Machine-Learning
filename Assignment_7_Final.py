######################################Assignment- 7  #######################################
################################### Mahendra Thakur #####################################


from urllib.request import urlopen as uopen
#import the Beautiful soup functions to parse the data returned from the website
from bs4 import BeautifulSoup as soup

#specify the url
cricbuzz = "http://www.cricbuzz.com/cricket-team/india/2/schedule"

#Query the website and return the html to the variable 'page'
client = uopen(cricbuzz)
page_html=client.read()
client.close()

#parsing HTML

page_soup=soup(page_html,"html.parser")
page_soup.prettify()

datecontainer=page_soup.findAll("div",{"class":"cb-col-100 cb-col cb-series-brdr "})
len(datecontainer)
datecontainer[0].span.string.strip()

matchcontainer=page_soup.findAll("div",{"class":"cb-col-75 cb-col"})
len(matchcontainer)
matchcontainer[0].span.string.strip() +' '+ matchcontainer[0].div.div.string.strip()

venuecontainer=page_soup.findAll("div",{"class":"text-gray cb-ovr-flo"})
len(venuecontainer)
venuecontainer[0].text.strip()

timecontainer=page_soup.findAll("div",{"class":"cb-font-12 text-gray"})
len(timecontainer)
timecontainer[0].text[16:24].strip()+';'+ timecontainer[0].text.strip()


#Making Columns - using list comprehension

Date = [ datecontainer[i].span.string.strip() for i in range(0,len(datecontainer)) ]
Match_Details = [ matchcontainer[i].span.string.strip() for i in range(1,len(matchcontainer)) ]
Venue = [ venuecontainer[i].text.strip() for i in range(1,len(venuecontainer)) ]
Time = [ timecontainer[i].text[16:24].strip() +';'+ timecontainer[i].text.strip() for i in range(1,len(timecontainer)) ]


import pandas as pd
cricbuzzdf = pd.DataFrame({
        "Date": Date, 
        "Match Details": Match_Details, 
        "Venue": Venue, 
        "Time": Time
    })

#Reordering Columns

cricbuzzdf = cricbuzzdf[["Date", "Match Details", "Venue", "Time"]]
cricbuzzdf

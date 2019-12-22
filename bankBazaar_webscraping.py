import bs4 as bs
from urllib.request import Request, urlopen
import pandas as pd
url = 'https://www.bankbazaar.com/reviews/hdfc-limited/home-loan.html?reviewPageNumber='
page= 1
comments = []
##issue with 404, check links
ratings = []
links = ['https://www.bankbazaar.com/reviews/karnataka-bank/home-loan.html?reviewPageNumber=',
         'https://www.bankbazaar.com/reviews/pnb-housing-finance-limited/home-loan.html?reviewPageNumber='
         'https://www.bankbazaar.com/reviews/kotak-mahindra-bank/home-loan.html?reviewPageNumber=',
         'https://www.bankbazaar.com/reviews/hdfc-limited/home-loan.html?reviewPageNumber=']
for i in links:
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    for page in range(1,2): # fix this 1 to 51 not 1 and 51 
        try:
            source = Request((str(i)+str(page)), headers={'User-Agent': 'Mozilla/7.0'})
            webpage = urlopen(source).read()
            soup = bs.BeautifulSoup(webpage,'lxml')
            for item in soup.find_all('div',class_ = 'text_here review-desc-more'):
                comments.append(str(item.text))
                print('entered loop 1')

            for rating in soup.find_all('div', class_='rating-section review-user-score'):
                ratings.append(str(rating.text))
                print('entered loop 2')

            print("loop"+" "+ str(page))
        except:
            continue
        
    df1 = pd.DataFrame({'Comments':comments})
    df2 = pd.DataFrame({'Ratings':ratings})
    df3 = pd.concat([df1,df2],axis=1, sort =False)

    print(df3.tail())
    df3.to_excel('C:/Users/rochisman.datta/Desktop/Python Code/{}.xlsx'.format(i[34:40]))


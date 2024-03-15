# from dynamic_pricing.price import Price
from dynamic_pricing.scrape import scrape

from flask import Flask
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)


# @app.route('/pricing', methods=['GET'])
# def pricing():
#     price = Price()
#     return "300 euro voor een banaan"

@app.route('/scrape-data', methods=['GET'])
def pricing():
    scrape()
    return "Data scraped!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
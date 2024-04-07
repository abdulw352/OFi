from flask import Flask, request, jsonify
from trading_bot import BidderBot, SellerBot

app = Flask(__name__)

# Dictionary to store active negotiations
active_negotiations = {}

@app.route('/negotiate', methods=['POST'])
def negotiate_price():
    data = request.get_json()
    user_id = data['user_id']
    user_role = data['user_role']  # 'bidder' or 'seller'
    initial_price = data['initial_price']
    reservation_price = data['reservation_price']

    # Check if the user is already in an active negotiation
    if user_id in active_negotiations:
        return jsonify({'error': 'You are already in an active negotiation.'}), 400

    if user_role == 'bidder':
        bot = BidderBot(initial_price, reservation_price)
    else:
        bot = SellerBot(initial_price, reservation_price)

    active_negotiations[user_id] = bot

    return jsonify({'message': 'Negotiation started. Wait for the other party to respond.'})

@app.route('/offer', methods=['POST'])
def make_offer():
    data = request.get_json()
    user_id = data['user_id']
    offer = data['offer']

    if user_id not in active_negotiations:
        return jsonify({'error': 'You are not in an active negotiation.'}), 400

    bot = active_negotiations[user_id]
    if isinstance(bot, BidderBot):
        agreed_price = bot.negotiate(offer)
    else:
        agreed_price = bot.negotiate(offer)

    if agreed_price is not None:
        del active_negotiations[user_id]
        return jsonify({'agreed_price': agreed_price})
    else:
        return jsonify({'message': 'Negotiation failed. No agreement reached.'})

if __name__ == '__main__':
    app.run(debug=True)

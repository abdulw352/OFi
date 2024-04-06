import time
import random

# Parameters for the Rubinstein bargaining model
DISCOUNT_FACTOR_BIDDER = 0.9
DISCOUNT_FACTOR_SELLER = 0.8
MAX_ITERATIONS = 10

class BidderBot:
    def __init__(self, initial_bid, reservation_price):
        self.initial_bid = initial_bid
        self.reservation_price = reservation_price
        self.current_offer = initial_bid

    def negotiate(self, seller_ask_price):
        iterations = 0
        while iterations < MAX_ITERATIONS:
            iterations += 1
            print(f"Bidder: Offering {self.current_offer}")
            time.sleep(1)

            # Calculate the new offer using the Rubinstein bargaining model
            self.current_offer = max(self.reservation_price, seller_ask_price * DISCOUNT_FACTOR_BIDDER)

            # Check if the offer is acceptable to the seller
            if self.current_offer >= seller_ask_price:
                print(f"Bidder: Offer accepted at {self.current_offer}")
                return self.current_offer
            else:
                print(f"Bidder: Offer rejected. Waiting for seller's response.")
                time.sleep(random.uniform(1, 3))

        print("Bidder: Couldn't reach an agreement. Negotiations failed.")
        return None

class SellerBot:
    def __init__(self, initial_ask, reservation_price):
        self.initial_ask = initial_ask
        self.reservation_price = reservation_price
        self.current_offer = initial_ask

    def negotiate(self, bidder_offer):
        iterations = 0
        while iterations < MAX_ITERATIONS:
            iterations += 1
            print(f"Seller: Asking {self.current_offer}")
            time.sleep(1)

            # Calculate the new offer using the Rubinstein bargaining model
            self.current_offer = min(self.reservation_price, bidder_offer * DISCOUNT_FACTOR_SELLER)

            # Check if the offer is acceptable to the bidder
            if self.current_offer <= bidder_offer:
                print(f"Seller: Offer accepted at {self.current_offer}")
                return self.current_offer
            else:
                print(f"Seller: Offer rejected. Waiting for bidder's response.")
                time.sleep(random.uniform(1, 3))

        print("Seller: Couldn't reach an agreement. Negotiations failed.")
        return None

# Example usage
bidder = BidderBot(initial_bid=50, reservation_price=40)
seller = SellerBot(initial_ask=80, reservation_price=60)

# Start the negotiation
agreed_price = bidder.negotiate(seller.initial_ask)
if agreed_price is not None:
    print(f"Agreement reached at price: {agreed_price}")
else:
    print("No agreement reached.")

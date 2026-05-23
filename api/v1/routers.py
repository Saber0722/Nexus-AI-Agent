from flask import Flask, request, jsonify
from models import Cart, Product

app = Flask(__name__)

@app.route('/cart', methods=['POST'])
def add_product_to_cart():
    cart_id = request.json.get('cartId')
    product_id = request.json.get('productId')

    if not cart_id or not product_id:
        return jsonify({"error": "Cart ID and Product ID are required"}), 400

    cart = Cart.query.get(cart_id)
    if not cart:
        return jsonify({"error": "Cart not found"}), 404

    product = Product.query.get(product_id)
    if not product:
        return jsonify({"error": "Product not found"}), 404

    # Assuming the product can be added to the cart
    cart.products.append(product)
    cart.save()

    return jsonify({"message": f"Product {product_id} added to cart {cart_id}"}), 201

if __name__ == '__main__':
    app.run(debug=True)
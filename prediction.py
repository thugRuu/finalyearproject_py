import random
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize the Flask app
app = Flask(__name__)

# Load the model
loaded_model = joblib.load('random_forest_model.joblib')

# Define the function to suggest alternatives based on user data
def suggest_alternatives(user_data):
    suggestions = []
    diet_suggestions = [
        "Consider adopting a plant-based diet such as vegetarian or vegan to reduce your carbon footprint.",
        "If you enjoy cooking, try experimenting with more plant-based recipes.",
        "Consider meal prepping with sustainable ingredients to minimize food waste."
    ]
    if user_data['diet'] == 'omnivore':
        suggestions.append(random.choice(diet_suggestions))
    elif user_data['diet'] == 'pescatarian':
        suggestions.append("Reducing seafood consumption and transitioning to a vegetarian or vegan diet can further reduce carbon emissions.")
    
    # Shower frequency suggestion
    shower_suggestions = [
        "Reduce shower frequency to once a day or less to save water and energy.",
        "Consider taking shorter showers and using a water-efficient showerhead.",
        "Try alternating between showering and bathing to conserve water."
    ]
    if user_data['showerFrequency'] == 'twice a day':
        suggestions.append(random.choice(shower_suggestions))

    # heatingSource suggestion
    heating_suggestions = [
        "Switch to cleaner heating options such as natural gas, electricity, or renewable sources like solar.",
        "Install a smart thermostat to optimize heating efficiency.",
        "Consider insulating your home to reduce heating needs."
    ]
    if user_data['heatingSource'] == 'coal':
        suggestions.append(random.choice(heating_suggestions))
    elif user_data['heatingSource'] == 'natural gas':
        suggestions.append("Consider switching to electric heating or renewable energy sources to further reduce emissions.")
    
    # Transport and Vehicle Type suggestion
    transport_suggestions = [
        "Use a bicycle or walk for shorter distances to minimize emissions.",
        "Carpool with friends or colleagues to reduce your carbon footprint.",
        "Explore public transport options like buses and trains for longer distances."
    ]
    if user_data['transportation'] == 'private' and user_data['monthlyDistance'] <= 200:
        suggestions.append(random.choice(transport_suggestions))
    elif user_data['transportation'] == 'private' and user_data['vehicleType'] in ['petrol', 'diesel']:
        suggestions.append("Switch to a hybrid or electric vehicle to reduce carbon emissions.")
    
    # socialActivity suggestion
    social_activity_suggestions = [
        "Consider balancing social activities with eco-friendly options such as virtual meetups.",
        "Host gatherings at home to reduce travel and emissions.",
        "Engage in outdoor activities that minimize transportation."
    ]
    if user_data['socialActivity'] == 'often':
        suggestions.append(random.choice(social_activity_suggestions))
    
    # Monthly Grocery Bill suggestion
    grocery_suggestions = [
        "Opt for locally sourced, seasonal, and plant-based foods to reduce the carbon footprint of your groceries.",
        "Plan your meals ahead to avoid unnecessary grocery purchases.",
        "Buy in bulk to minimize packaging waste."
    ]
    if user_data['groceryBill'] > 300:
        suggestions.append(random.choice(grocery_suggestions))

    # travelFrequency suggestion
    air_travel_suggestions = [
        "Reduce air travel and explore alternatives like trains or participate in carbon offset programs.",
        "Combine trips to minimize air travel frequency.",
        "Use video conferencing for business meetings to cut down on travel."
    ]
    if user_data['travelFrequency'] in ['frequently', 'very frequently']:
        suggestions.append(random.choice(air_travel_suggestions))

    # Vehicle Monthly Distance suggestion
    vehicle_distance_suggestions = [
        "Reduce long car journeys and consider public transport or carpooling for long distances.",
        "Plan your routes to minimize driving distance.",
        "Consider telecommuting to reduce the need for daily travel."
    ]
    if user_data['monthlyDistance'] > 1000:
        suggestions.append(random.choice(vehicle_distance_suggestions))

    # wasteBagSize and Weekly Count suggestion
    waste_suggestions = [
        "Reduce waste by recycling more, composting, and opting for reusable items to minimize trash volume.",
        "Consider reducing packaging by choosing bulk items.",
        "Participate in community clean-up days to raise awareness about waste."
    ]
    if user_data['wasteBagSize'] in ['large', 'extra large']:
        suggestions.append(random.choice(waste_suggestions))
    if user_data['wasteBagCount'] > 2:
        suggestions.append("Reduce your waste output by composting organic waste and recycling more frequently.")

    # TV/PC Daily Usage suggestion
    screen_time_suggestions = [
        "Reduce screen time and save energy by limiting non-essential TV or PC usage.",
        "Engage in outdoor activities instead of spending too much time on screens.",
        "Use energy-efficient devices to minimize power consumption."
    ]
    if user_data['tvComputerHours'] > 4:
        suggestions.append(random.choice(screen_time_suggestions))

    # New Clothes Monthly suggestion
    clothing_suggestions = [
        "Try to reduce clothes shopping or purchase second-hand or eco-friendly brands to lower your carbon footprint.",
        "Host a clothing swap with friends to refresh your wardrobe sustainably.",
        "Support local artisans and sustainable fashion brands."
    ]
    if user_data['newClothesMonthly'] > 2:
        suggestions.append(random.choice(clothing_suggestions))

    # Internet Daily Usage suggestion
    internet_usage_suggestions = [
        "Consider reducing internet usage or switching to energy-efficient devices to cut down on energy consumption.",
        "Use internet time for productive activities like learning a new skill.",
        "Limit streaming services to reduce energy usage."
    ]
    if user_data['internetHours'] > 4:
        suggestions.append(random.choice(internet_usage_suggestions))

    # Energy efficiency suggestion
    energy_efficiency_suggestions = [
        "Improve home energy efficiency by using LED bulbs, energy-efficient appliances, and insulating your home.",
        "Conduct an energy audit to identify areas for improvement.",
        "Consider renewable energy options such as solar panels."
    ]
    if user_data['energyEfficiency'] == 2:
        suggestions.append(random.choice(energy_efficiency_suggestions))

    # Shuffle suggestions to randomize the order
    random.shuffle(suggestions)


    return suggestions

# API route to accept data and return predictions and suggestions
@app.route('/predict', methods=['POST'])
def predict_and_suggest():
    data = request.json
   
    new_data = pd.DataFrame([data], columns=[
    "diet",
    "showerFrequency",
    "heatingSource",
    "transportation",
    "vehicleType",
    "socialActivity",
    "groceryBill",
    "travelFrequency",
    "monthlyDistance",
    "wasteBagSize",
    "wasteBagCount",
    "tvComputerHours",
    "newClothesMonthly",
    "internetHours",
    "energyEfficiency"])

    # Make prediction
    carbon_footprint = loaded_model.predict(new_data)[0]

    # Map numerical data to readable values for suggestions
    user_data = user_data = {
    'diet': ['pescatarian', 'vegetarian', 'omnivore', 'vegan'][int(new_data['diet'][0])],
    'showerFrequency': ['daily', 'less frequently', 'more frequently', 'twice a day'][int(new_data['showerFrequency'][0])],
    'heatingSource': ['coal', 'natural gas', 'wood', 'electricity'][int(new_data['heatingSource'][0])],
    'transportation': ['public', 'walk/bicycle', 'private'][int(new_data['transportation'][0])],
    'vehicleType': ['None', 'petrol', 'diesel', 'hybrid', 'lpg', 'electric'][int(new_data['vehicleType'][0])],
    'socialActivity': ['often', 'never', 'sometimes'][int(new_data['socialActivity'][0])],
    'groceryBill': int(new_data['groceryBill'][0]),  # Convert to integer
    'travelFrequency': ['frequently', 'rarely', 'never', 'very frequently'][int(new_data['travelFrequency'][0])],
    'monthlyDistance': int(new_data['monthlyDistance'][0]),
    'wasteBagSize': ['small', 'medium', 'large', 'extra large'][int(new_data['wasteBagSize'][0])],
    'wasteBagCount': int(new_data['wasteBagCount'][0]),
    'tvComputerHours': int(new_data['tvComputerHours'][0]),
    'newClothesMonthly': int(new_data['newClothesMonthly'][0]),
    'internetHours': int(new_data['internetHours'][0]),
    'energyEfficiency': ['Yes', 'Sometimes', 'No'][int(new_data['energyEfficiency'][0])]
}


    # Get suggestions based on user data
    alternative_suggestions = suggest_alternatives(user_data)

    # Return response as JSON
    return jsonify({
        'carbon_footprint': carbon_footprint,
        'suggestions': alternative_suggestions
    })

if __name__ == '__main__':
    app.run(debug=True)

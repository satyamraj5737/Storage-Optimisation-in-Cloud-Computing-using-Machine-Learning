from flask_cors import CORS
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json
from flask_restful import Api, Resource
from flasgger import Swagger
import matplotlib
import os

matplotlib.use('Agg')

app = Flask(__name__)
CORS(app)
api = Api(app)
Swagger(app)

# Load the trained model and features
loaded_classifier = joblib.load('model/decision_tree_model.joblib')
loaded_features = joblib.load('model/model_features.joblib')


def calculate_right_sized(property) -> int:
    property = int(property)
    if property == 1:
        return 3
    return property // 2 + property // 4


def map_value(value, from_min, from_max, to_min, to_max):
    # Ensure the value is within the source range
    value = max(min(value, from_max), from_min)

    # Map the value from the source range to the destination range
    mapped_value = (value - from_min) / (from_max -
                                         from_min) * (to_max - to_min) + to_min

    return round(mapped_value, 3)


def move_to_various_cloud(row):
    cpu_utilization = row[5]
    latency = row[6]
    if row[1] == "On-prem":
        return [
            1.0,
            1.0 + map_value(cpu_utilization % 10, 0, 9, -0.3, 0.3),
            1.0 + map_value(cpu_utilization % 10, 0, 9, -0.3, 0.3) +
            map_value(latency % 10, 0, 9, -0.3, 0.3),
        ]
    if row[1] == "AWS":
        return [
            1.0 + map_value(cpu_utilization % 10, 0, 9, -0.3, 0.3),
            1.0,
            1.0 + map_value(cpu_utilization % 10, 0, 9, -0.3, 0.3) +
            map_value(latency % 10, 0, 9, -0.3, 0.3),
        ]
    if row[1] == "GCP":
        return [
            1.0 + map_value(cpu_utilization % 10, 0, 9, -0.3, 0.3),
            1.0 + map_value(cpu_utilization % 10, 0, 9, -0.3, 0.3) +
            map_value(latency % 10, 0, 9, -0.3, 0.3),
            1.0,
        ]


def movement_related_calculation(row):
    move_onprem = row[10]
    cpu_utilization = row[5]
    cost = row[7]
    latency = row[6]
    move_aws = row[11]
    move_gcp = row[12]
    updated_latency = int(latency) + 
        map_value(cpu_utilization, 0, 100, -20, 20)
    min_val = min(float(move_onprem), float(move_aws), float(move_gcp))
    if min_val < 0.84:
        if min_val == float(move_onprem):
            return ['On-prem', float(cost) * min_val, int(updated_latency)]
        elif min_val == float(move_aws):
            return ['AWS', float(cost) * min_val, int(updated_latency)]
        else:
            return ['GCP', float(cost) * min_val, int(updated_latency)]

    return ['No movement required', cost, latency]


def predict_using_model(input_csv_path):
    # Load the input CSV file
    input_data = pd.read_csv(input_csv_path)

    # Select relevant features from the input datasheet
    X_input = input_data[loaded_features]

    # Make predictions using the loaded model
    predictions = loaded_classifier.predict(X_input)

    # Add predictions to the input datasheet
    input_data['Predictions'] = predictions

    # create columns for movement logic
    moved_data = input_data.apply(
        lambda row: move_to_various_cloud(row), axis=1)
    input_data[['move_onprem', 'move_aws', 'move_gcp']] = pd.DataFrame(
        moved_data.tolist(), index=input_data.index)

    # create columns for movement-related calculation
    moved_data = input_data.apply(
        lambda row: movement_related_calculation(row), axis=1)
    input_data[['where to move', 'Updated Cost', 'Updated latency']
               ] = pd.DataFrame(moved_data.tolist(), index=input_data.index)

    # Add columns for updated RAM and updated CPU
    input_data['Updated Ram'] = input_data.apply(
        lambda row: calculate_right_sized(row['Total Ram']) if row['Predictions'] == "underutilized" and int(row['Total Ram']) > 1 else (calculate_right_sized(row['Total Ram']) // 3 + int(row['Total Ram'])) if float(row['Cpu Utilization']) > 94 else row['Total Ram'], axis=1)

    input_data['Updated CPU'] = input_data.apply(
        lambda row: calculate_right_sized(row['Total CPU']) if row['Predictions'] == "underutilized" and int(row['Total CPU']) > 1 else (calculate_right_sized(row['Total CPU']) // 3 + int(row['Total CPU'])) if float(row['Cpu Utilization']) > 94 else row['Total CPU'], axis=1)

    # Save the datasheet with predictions to a new CSV file
    output_csv_path = 'datasets/predicted.csv'
    input_data.to_csv(output_csv_path, index=False)

    return output_csv_path


def get_count_of_cloud(data):
    cloud_count = {}
    cloud_cost = {}

    for index, row in data.iterrows():
        cloud_provider = row['cloud provider']
        total_cost = float(row['Cost Per Year ($)'])

        # Initialize count and cost for the cloud provider if not present
        if cloud_provider not in cloud_count:
            cloud_count[cloud_provider] = 0
            cloud_cost[cloud_provider] = 0.0

        # Increment the count and add to the total cost
        cloud_count[cloud_provider] += 1
        cloud_cost[cloud_provider] += total_cost

    # Round the total cost to 2 digits
    rounded_cloud_cost = {provider: round(
        cost, 2) for provider, cost in cloud_cost.items()}

    # Create a dictionary to hold the result
    result = {'cloud_count': cloud_count,
              'cloud_total_cost': rounded_cloud_cost}
    return result


def top_filter_vm_based_on_cloud(data, cloud_type):
    return json.loads(json.dumps(data[data['cloud provider'] == cloud_type].head(5).to_dict(orient='records')))


def filter_vm_based_on_cloud(data):
    result = []
    for _data in data:
        if _data['where to move'] != "No movement required":
            result.append(
                f'VM id={_data["id"]} can be moved to {_data["where to move"]} ')

    return result


def filter_vm_based_on_CPU_RAM_changes(file_path, cloud_type):
    data = pd.read_csv(file_path)
    filterd_data = data[data['cloud provider']
                        == cloud_type].to_dict(orient='records')
    result = []
    for _data in filterd_data:
        if _data['Total Ram'] != _data["Updated Ram"]:
            _data['Updated Cost'] = int(
                _data['Cost Per Year ($)'])*(float(_data["Updated Ram"])/_data['Total Ram'])
            result.append(_data)

    return result


def right_sizing_related_data(data):
    result = []
    for _data in data:
        if _data['Predictions'] == "underutilized":
            if int(_data['Total Ram']) > 1:
                result.append(
                    f'VM id={_data["id"]} reduce Ram to {calculate_right_sized(_data["Total Ram"])} ')

            if int(_data['Total CPU']) > 1:
                result.append(
                    f'VM id={_data["id"]} reduce CPU to {calculate_right_sized(_data["Total CPU"])} ')

        if float(_data['Cpu Utilization']) > 94:
            result.append(
                f'VM id={_data["id"]} increase CPU to {calculate_right_sized(_data["Total CPU"])//3 +int(_data["Total CPU"])} ')
            result.append(
                f'VM id={_data["id"]} increase Ram to {calculate_right_sized(_data["Total Ram"])//3 +int(_data["Total Ram"])} ')

    return result


def get_graph_data(file_path, cloud_type):
    data = pd.read_csv(file_path)
    filterd_data = data[data['cloud provider'] == cloud_type]

    # Convert the filtered data to a DataFrame
    filterd_data_df = pd.DataFrame(filterd_data)

    # Calculate the sum of 'Cost Per Year ($)' for the filtered data
    filtered_total_cost_sum = filterd_data_df['Cost Per Year ($)'].sum()

    # Calculate the sum of 'Updated Cost' for the filtered data
    filtered_updated_cost_moved = filterd_data_df['Updated Cost'].sum()

    # Calculate the sum of 'Updated Cost' after Resizing
    filtered_updated_right_sized_cost = 0
    for index, _data in filterd_data_df.iterrows():
        if _data['Total Ram'] != _data['Updated Ram']:
            _data['Updated Cost'] = int(
                _data['Cost Per Year ($)']) * (int(_data['Updated Ram']) / int(_data['Total Ram']))
            filtered_updated_right_sized_cost = filtered_updated_right_sized_cost + \
                _data['Updated Cost']
        else:
            filtered_updated_right_sized_cost = filtered_updated_right_sized_cost + \
                int(_data['Cost Per Year ($)'])

    return {'filtered_total_cost_sum': int(filtered_total_cost_sum), 'filtered_updated_cost_moved': int(filtered_updated_cost_moved*0.68), 'filtered_updated_right_sized_cost': int(filtered_updated_right_sized_cost*0.8)}


class PredictResource(Resource):
    def post(self):
        """
        Endpoint to perform predictions using the trained model.

        ---
        parameters:
          - name: file
            in: formData
            type: file
            required: false  # Make the file parameter optional
            description: The CSV file for prediction.
        responses:
          200:
            description: Success response with first page data.
          400:
            description: Error response if the file is not valid.
        """
        # Check if a file is provided in the request
        if 'file' not in request.files or not request.files['file']:
            # No file provided, use the dummy file
            file_path = os.path.join(os.path.dirname(__file__), 'test_dataset.csv')
        else:
            file = request.files['file']

            # Check if the file is of CSV format
            if file and file.filename.endswith('.csv'):
                # Save the uploaded CSV file
                file_path = 'uploads/' + file.filename
                file.save(file_path)

            else:
                return jsonify({'error': 'Please upload a valid CSV file'})

        # Perform analysis using the trained model
        # Read the predicted CSV file content
        data = pd.read_csv(predict_using_model(file_path))

        first_page_data = get_count_of_cloud(data)

        # Send both the generated graphs and predictions as a response
        return jsonify({'success': True,
                        'first_page_data': first_page_data
                        })

api.add_resource(PredictResource, '/api/predict')


def get_cloud_movement_data(file_path, cloud_type):
    data = pd.read_csv(file_path)
    return top_filter_vm_based_on_cloud(data, cloud_type)


def get_suggestions_list(file_path, cloud_type):
    data = pd.read_csv(file_path)
    filterd_data = data[data['cloud provider']
                        == cloud_type].to_dict(orient='records')
    suggestion_list = {
        'moved_vm_data': filter_vm_based_on_cloud(filterd_data),
        'right_sized_vm': right_sizing_related_data(filterd_data),
    }
    return suggestion_list


@app.route('/api/predict/aws', methods=['GET'])
def get_saved_predictions_aws():
    """
    Endpoint to get saved predictions for AWS.

    ---
    responses:
      200:
        description: Success response with AWS-specific prediction data.
      404:
        description: Error response if data for AWS is not found.
    """
    file_path = 'datasets/predicted.csv'
    cloud_type = 'AWS'
    predictions_data = get_cloud_movement_data(file_path, cloud_type)

    if not predictions_data:
        return jsonify({'error': f'No data found for {cloud_type}'})

    return jsonify({'success': True, 'cloud_type': cloud_type, 'cloud_movement_table': predictions_data})


@app.route('/api/predict/on_prem', methods=['GET'])
def get_saved_predictions_on_prem():
    """
    Endpoint to get saved predictions for On-prem.

    ---
    responses:
      200:
        description: Success response with On-prem-specific prediction data.
      404:
        description: Error response if data for On-prem is not found.
    """
    file_path = 'datasets/predicted.csv'
    cloud_type = 'On-prem'
    predictions_data = get_cloud_movement_data(file_path, cloud_type)

    if not predictions_data:
        return jsonify({'error': f'No data found for {cloud_type}'})

    return jsonify({'success': True, 'cloud_type': cloud_type, 'cloud_movement_table': predictions_data})


@app.route('/api/predict/gcp', methods=['GET'])
def get_saved_predictions_gcp():
    """
    Endpoint to get saved predictions for GCP.

    ---
    responses:
      200:
        description: Success response with GCP-specific prediction data.
      404:
        description: Error response if data for GCP is not found.
    """
    file_path = 'datasets/predicted.csv'
    cloud_type = 'GCP'
    predictions_data = get_cloud_movement_data(file_path, cloud_type)

    if not predictions_data:
        return jsonify({'error': f'No data found for {cloud_type}'})

    return jsonify({'success': True, 'cloud_type': cloud_type, 'cloud_movement_table': predictions_data})


@app.route('/api/predict/aws/right_size', methods=['GET'])
def get_saved_predictions_aws_right_size():
    """
    Endpoint to get saved predictions for AWS.

    ---
    responses:
      200:
        description: Success response with AWS-specific prediction data.
      404:
        description: Error response if data for AWS is not found.
    """
    file_path = 'datasets/predicted.csv'
    cloud_type = 'AWS'
    suggestions_list = get_suggestions_list(file_path, cloud_type)
    cloud_right_size_table = filter_vm_based_on_CPU_RAM_changes(
        file_path, cloud_type)

    graph_data = get_graph_data(file_path, cloud_type)
    if not suggestions_list:
        return jsonify({'error': f'No data found for {cloud_type}'})

    return jsonify({'success': True, 'cloud_type': cloud_type, 'cloud_right_size_table': cloud_right_size_table, 'suggestions_list': suggestions_list, 'graph_data': graph_data})


@app.route('/api/predict/on_prem/right_size', methods=['GET'])
def get_saved_predictions_on_prem_right_size():
    """
    Endpoint to get saved predictions for On-prem.

    ---
    responses:
      200:
        description: Success response with On-prem-specific prediction data.
      404:
        description: Error response if data for On-prem is not found.
    """
    file_path = 'datasets/predicted.csv'
    cloud_type = 'On-prem'
    suggestions_list = get_suggestions_list(file_path, cloud_type)
    cloud_right_size_table = filter_vm_based_on_CPU_RAM_changes(
        file_path, cloud_type)

    graph_data = get_graph_data(file_path, cloud_type)
    if not suggestions_list:
        return jsonify({'error': f'No data found for {cloud_type}'})

    return jsonify({'success': True, 'cloud_type': cloud_type, 'cloud_right_size_table': cloud_right_size_table, 'suggestions_list': suggestions_list, 'graph_data': graph_data})


@app.route('/api/predict/gcp/right_size', methods=['GET'])
def get_saved_predictions_gcp_right_size():
    """
    Endpoint to get saved predictions for GCP.

    ---
    responses:
      200:
        description: Success response with GCP-specific prediction data.
      404:
        description: Error response if data for GCP is not found.
    """
    file_path = 'datasets/predicted.csv'
    cloud_type = 'GCP'
    suggestions_list = get_suggestions_list(file_path, cloud_type)
    cloud_right_size_table = filter_vm_based_on_CPU_RAM_changes(
        file_path, cloud_type)

    graph_data = get_graph_data(file_path, cloud_type)
    if not suggestions_list:
        return jsonify({'error': f'No data found for {cloud_type}'})

    return jsonify({'success': True, 'cloud_type': cloud_type, 'cloud_right_size_table': cloud_right_size_table, 'suggestions_list': suggestions_list, 'graph_data': graph_data})


if __name__ == '__main__':
    app.run(port=3500, debug=True)

import base64

import numpy as np
import pandas as pd

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import random

import streamlit as st
from streamlit_dynamic_filters import DynamicFilters


def generate_matrix(matrix, selected_indices):
    selected_rows = selected_indices
    selected_columns = selected_indices
    return matrix[selected_rows][:, selected_columns]


def create_data_model(combo, extra_time, start):
    """Stores the data for the problem."""
    duration_matrix = pd.read_csv('duration_matrix_csv_1.csv', header=None)
    numpy_array = duration_matrix.values
    distance_matrix_car = generate_matrix(numpy_array, combo)
    print(len(distance_matrix_car))
    # Update the matrix a based on extra_time
    for entry in extra_time:
        try:
            i, time = combo.index(entry[0]), entry[1]
            for j in range(len(distance_matrix_car)):
                distance_matrix_car[i][j] += time
                distance_matrix_car[j][i] += time
        except:
            pass
    data = {}
    data["distance_matrix"] = distance_matrix_car
    data["num_vehicles"] = 1
    data["starts"] = [start]
    data["ends"] = [0]
    data["depot"] = 0
    return data


def create_data_model_priority(combo, extra_time, start):
    """Stores the data for the problem."""
    duration_matrix = pd.read_csv('duration_matrix_csv_1.csv', header=None)
    numpy_array = duration_matrix.values
    distance_matrix_car = generate_matrix(numpy_array, combo)
    len_distance_matrix_car = len(distance_matrix_car)
    # Update the matrix a based on extra_time
    for entry in extra_time:
        try:
            i, time = combo.index(entry[0]), entry[1]
            for j in range(len(distance_matrix_car)):
                distance_matrix_car[i][j] += time
                distance_matrix_car[j][i] += time
        except:
            pass
    for j in range(len(distance_matrix_car)):
        distance_matrix_car[j][0] = 0
    data = {}
    data["distance_matrix"] = distance_matrix_car
    data["num_vehicles"] = 1
    data["starts"] = [start]
    data["ends"] = [0]
    data["depot"] = 0
    return data


def print_solution(data, manager, routing, solution, combo):
    """Prints solution on console."""
    max_route_distance = 0
    route_1 = []
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle:\n"
        route_distance = 0
        while not routing.IsEnd(index):
            route_1.append(combo[manager.IndexToNode(index)])
            plan_output += f" {combo[manager.IndexToNode(index)]} -> "
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        plan_output += f"{combo[manager.IndexToNode(index)]}\n"
        plan_output += f"Distance of the route: {route_distance}\n"
        print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
    # print(f"Maximum of the route distances: {max_route_distance}m")
    return route_1, route_distance


def main(data):
    """Entry point of the program."""
    # Instantiate the data problem.
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']),
        data['num_vehicles'],
        data['starts'],
        data['ends'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = "Distance"
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        1200,  # vehicle maximum travel distance
        False,  # start cumul to zero
        dimension_name,
    )

    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)
    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    # Print solution on console.
    return data, manager, routing, solution


def main_route(route, extra_time):
    # Gọi hàm main với các tham số route và extra_time
    main(route, extra_time)


def modify_list(lst):
    if 0 in lst:
        lst = [i for i in lst if i != 0]
    lst.insert(0, 0)
    return lst


@st.cache_data
def load_data():
    list_atm = random.sample(range(41), 20)
    return list_atm


def call_main():
    st.title("Các cây ATM cần tiếp")

    list_atm = load_data()

    # Tạo chuỗi HTML từ danh sách
    html = "<p style='text-align: center;'>"
    for number in list_atm:
        html += str(number) + " &nbsp; "
    html += "</p>"

    # Hiển thị chuỗi HTML
    st.markdown(html, unsafe_allow_html=True)

    xe_1 = []
    xe_2 = []
    # Tạo một danh sách 20 số
    for i in list_atm:
        if i <= 20:
            xe_1.append(i)
        else:
            xe_2.append(i)
    xe_1.sort()
    xe_2.sort()
    # Create a dataframe
    data = {
        'Xe phụ trách': ['xe 1', 'xe 2'],
        'Danh sách ATM': [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                          [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]],
        'Danh sách cây ATM cần tiếp': [xe_1, xe_2],
        'Cây ưu tiên tiếp': [xe_1[1:3], xe_2[1:3]]
    }
    df = pd.DataFrame(data)
    csv_data = df.to_csv(index=False)
    b64 = base64.b64encode(csv_data.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="data.csv" style="float: right;">Download CSV</a>'

    if st.button('Download'):
        st.markdown(href, unsafe_allow_html=True)
    # Display the dataframe
    html_table = df.to_html(escape=False)
    # Modify the HTML table to adjust row widths
    modified_html = html_table.replace('<th>', '<th style="text-align: center;">')
    modified_html = modified_html.replace('[', '').replace(']', '')
    # Display the modified HTML table
    st.markdown(modified_html, unsafe_allow_html=True)

    data = {
        'Type': ['DS sửa chữa', 'DS thay giấy', 'Khác'],
        'Pick_Option': [[], [], []],
        'Time': [[15], [5], [10]]
    }
    df_2 = pd.DataFrame(data)

    # Allow users to select options for the 'Pick Option' column
    for i in range(len(df_2)):
        pick_option = st.multiselect(f"{data['Type'][i]}", list_atm, df_2['Pick_Option'][i])
        df_2.at[i, 'Pick_Option'] = pick_option

    # Display the updated DataFrame
    html_table = df_2.to_html(escape=False)
    # Modify the HTML table to adjust row widths
    modified_html = html_table.replace('<th>', '<th style="text-align: center;">')
    modified_html = modified_html.replace('[', '').replace(']', '')
    # Display the modified HTML table
    st.markdown(modified_html, unsafe_allow_html=True)
    priority_1 = df['Cây ưu tiên tiếp'][0]
    priority_2 = df['Cây ưu tiên tiếp'][1]
    if st.button('Chạy'):
        data = {
            'Xe phụ trách': ['Xe_1', 'Xe_2'],
            'Route': [[], []],
            'Distance': [[], []]
        }
        df_3 = pd.DataFrame(data)
        extra_time = []
        if len(df_2['Pick_Option'][0]) > 0:
            for i in df_2['Pick_Option'][0]:
                extra_time.append([i, df_2['Time'][0][0]])
        if len(df_2['Pick_Option'][1]) > 0:
            for i in df_2['Pick_Option'][1]:
                extra_time.append([i, df_2['Time'][1][0]])
        if len(df_2['Pick_Option'][2]) > 0:
            for i in df_2['Pick_Option'][2]:
                extra_time.append([i, df_2['Time'][2][0]])
        if len(priority_1) > 0:
            priority_1 = modify_list(priority_1)
            xe_1 = modify_list(xe_1)
            data = create_data_model_priority(priority_1, extra_time, 0)
            data, manager, routing, solution = main(data)
            route_1, route_distance = print_solution(data, manager, routing, solution, priority_1)
            result = list(set(xe_1) - set(priority_1))
            result.insert(0, 0)
            result.insert(1, route_1[len(route_1) - 1])
            data = create_data_model(result, extra_time, 1)
            data, manager, routing, solution = main(data)
            route_2, route_distance_2 = print_solution(data, manager, routing, solution, result)
            df_3.at[0, 'Route'] = [route_1[0:len(route_1) - 1] + route_2]
            df_3.at[0, 'Distance'] = [route_distance + route_distance_2]
        else:
            xe_1 = modify_list(xe_1)
            data = create_data_model(xe_1, extra_time, 0)
            data, manager, routing, solution = main(data)
            route_1, route_distance = print_solution(data, manager, routing, solution, xe_1)
            df_3.at[0, 'Route'] = [route_1]
            df_3.at[0, 'Distance'] = [route_distance]
        if len(priority_2) > 0:
            priority_2 = modify_list(priority_2)
            xe_2 = modify_list(xe_2)
            data = create_data_model_priority(priority_2, extra_time, 0)
            data, manager, routing, solution = main(data)
            route_1, route_distance = print_solution(data, manager, routing, solution, priority_2)
            result = list(set(xe_2) - set(priority_2))
            result.insert(0, 0)
            result.insert(1, route_1[len(route_1) - 1])
            data = create_data_model(result, extra_time, 1)
            data, manager, routing, solution = main(data)
            route_2, route_distance_2 = print_solution(data, manager, routing, solution, result)
            df_3.at[1, 'Route'] = [route_1[0:len(route_1) - 1] + route_2]
            df_3.at[1, 'Distance'] = [route_distance + route_distance_2]
        else:
            xe_2 = modify_list(xe_2)
            data = create_data_model(xe_2, extra_time, 0)
            data, manager, routing, solution = main(data)
            route_1, route_distance = print_solution(data, manager, routing, solution, xe_2)
            df_3.at[1, 'Route'] = [route_1]
            df_3.at[1, 'Distance'] = [route_distance]

        # Display the updated DataFrame
        html_table = df_3.to_html(escape=False)
        # Modify the HTML table to adjust row widths
        modified_html = html_table.replace('<th>', '<th style="text-align: center;">')
        modified_html = modified_html.replace('[', '').replace(']', '')
        # Display the modified HTML table
        st.markdown(modified_html, unsafe_allow_html=True)


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    call_main()

# if __name__ == "__main__":
#     route = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
#     priority = [0, 1, 2]
#     extra_time = [[4, 15]]
#     data = create_data_model_priority(priority, extra_time, 0)
#     data, manager, routing, solution = main(data)
#     route_1, route_distance = print_solution(data, manager, routing, solution, priority)
#     result = list(set(route) - set(priority))
#     result.append(0)
#     print(result)
#     data = create_data_model(result, extra_time, route_1[len(route_1) - 1])
#     data, manager, routing, solution = main(data)
#     route_2, route_distance_2 = print_solution(data, manager, routing, solution, result)
#     print(route_1 + route_2)
#     print(route_distance + route_distance_2)

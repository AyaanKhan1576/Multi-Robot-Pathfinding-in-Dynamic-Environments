#%%
# Name: Ayaan Khan
# Roll Number: 22i-0832
# Section: CS-K
# AI Assignment 01

import re
import heapq
import random

class World:
    def __init__(self, grid, dynamic_agents, total_time):
        self.grid = grid
        self.dynamic_agents = dynamic_agents
        self.total_time = total_time
        self.N = len(grid)
        self.M = len(grid[0])
        self.time_map = self.build_time_map(dynamic_agents)
        
    def build_time_map(self, dynamic_agents):
        time_map = {}
        for agent in dynamic_agents.values():
            positions, times = agent["positions"], agent["times"]

            full_path = positions + positions[::-1][1:-1] 
            cycle_length = len(full_path)

            for t in range(self.total_time):
                cycle_time = t % cycle_length  
                pos = full_path[cycle_time]

                if t not in time_map:
                    time_map[t] = set()
                time_map[t].add(pos)

        return time_map

    
    def is_dynamic_obstacle(self, pos, time):
        if time in self.time_map:
            return pos in self.time_map[time]
        return False
    
    def in_bounds(self, pos):
        x, y = pos
        return 0 <= x < self.N and 0 <= y < self.M
    
    def is_static(self, pos):
        x, y = pos
        return self.grid[x][y] == 'X'
    
    def is_legal(self, pos, time):
        if not self.in_bounds(pos):
            return False
        if self.is_static(pos) or self.is_dynamic_obstacle(pos, time):
            return False
        return True


class Robot:
    def __init__(self, id, start_pos, goal_pos):
        self.id = id
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.current_pos = start_pos
        self.current_path = []
        self.path = []
        self.total_time = 0
        self.at_goal = False
        self.path_taken = []


class Parsing:
    def __init__(self, agents_file, robots_file, grid_file):
        self.agents_file = agents_file
        self.robots_file = robots_file
        self.grid_file = grid_file
        self.agents_data = self.parse_agents(agents_file)
        self.robots_data = self.parse_robots(robots_file)
        self.grid_data = self.parse_grid(grid_file)

    def parse_agents(self, file_path):
        agents = {}
        with open(file_path, 'r') as file:
            for line in file:
                agent_match = re.match(r"Agent (\d+):", line)
                if not agent_match:
                    continue
                agent_id = int(agent_match.group(1))
                positions_match = re.search(r"\[\(\((.*?)\)\)\]", line)
                if positions_match:
                    positions_str = positions_match.group(1)
                    positions = [tuple(map(int, p.split(','))) for p in positions_str.split("), (")]
                else:
                    positions = []
                times_match = re.search(r"at times \[(.*?)\]", line)
                if times_match:
                    times = list(map(int, times_match.group(1).split(", ")))
                else:
                    times = []
                agents[agent_id] = {"positions": positions, "times": times}
        return agents
    
    def parse_robots(self, file_path):
        robots = {}
        with open(file_path, 'r') as file:
            for line in file:
                match = re.match(r"Robot (\d+): Start \((\d+), (\d+)\) End \((\d+), (\d+)\)", line)
                if match:
                    robot_id = int(match.group(1))
                    start_pos = (int(match.group(2)), int(match.group(3)))
                    goal_pos = (int(match.group(4)), int(match.group(5)))
                    robots[robot_id] = Robot(robot_id, start_pos, goal_pos)
        return robots
    
    def parse_grid(self, file_path):
        grid = []
        with open(file_path, 'r') as f:
            size = int(f.readline().strip())
            for line in f:
                row = []
                for char in line:
                    if char == 'X':
                        row.append('X')
                    else:
                        row.append('.')
                grid.append(row)
        return grid


class Simulate:
    def __init__(self, robots, world):
        self.robots = robots
        self.world = world
        self.time = 0
        self.max_time = self.world.N * self.world.M

    def heuristic(self, pos, goal):
        distance = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        time_penalty = 0
        for t in range(1, 4):  
            if self.world.is_dynamic_obstacle(pos, self.time + t):
                time_penalty += 5  
        
        return distance + time_penalty


    def get_valid_moves(self, current_pos, time):
        moves = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (current_pos[0] + dx, current_pos[1] + dy)
            if self.world.is_legal(new_pos, time + 1):
                moves.append(new_pos)
        return moves

    def bidirectional_search(self, start, goal):
        if start == goal:
            return [start]

        open_f = []
        cost_f = {start: 0}
        parent_f = {start: None}
        heapq.heappush(open_f, (self.heuristic(start, goal), start))

        open_b = []
        cost_b = {goal: 0}
        parent_b = {goal: None}
        heapq.heappush(open_b, (self.heuristic(goal, start), goal))

        visited_f = set()
        visited_b = set()
        meeting_node = None
        best_cost = float('inf')

        while open_f and open_b:

            if open_f:
                _, current_f = heapq.heappop(open_f)
                visited_f.add(current_f)
                if current_f in visited_b:
                    total_cost = cost_f[current_f] + cost_b[current_f]
                    if total_cost < best_cost:
                        best_cost = total_cost
                        meeting_node = current_f

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    neighbor = (current_f[0] + dx, current_f[1] + dy)
                    if not self.world.in_bounds(neighbor) or self.world.is_static(neighbor):
                        continue
                    new_cost = cost_f[current_f] + 1

                    if neighbor not in cost_f or new_cost < cost_f[neighbor]:
                        cost_f[neighbor] = new_cost
                        parent_f[neighbor] = current_f
                        priority = new_cost + self.heuristic(neighbor, goal)
                        heapq.heappush(open_f, (priority, neighbor))

            if open_b:
                _, current_b = heapq.heappop(open_b)
                visited_b.add(current_b)
                if current_b in visited_f:
                    total_cost = cost_f[current_b] + cost_b[current_b]
                    if total_cost < best_cost:
                        best_cost = total_cost
                        meeting_node = current_b

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    neighbor = (current_b[0] + dx, current_b[1] + dy)
                    if not self.world.in_bounds(neighbor) or self.world.is_static(neighbor):
                        continue
                    new_cost = cost_b[current_b] + 1

                    if neighbor not in cost_b or new_cost < cost_b[neighbor]:
                        cost_b[neighbor] = new_cost
                        parent_b[neighbor] = current_b
                        priority = new_cost + self.heuristic(neighbor, start)
                        heapq.heappush(open_b, (priority, neighbor))

            if meeting_node is not None:
                min_f = open_f[0][0] if open_f else float('inf')
                min_b = open_b[0][0] if open_b else float('inf')
                if best_cost <= min_f + min_b:
                    break

        if meeting_node is None:
            return None

        path_forward = []
        node = meeting_node
        while node is not None:
            path_forward.append(node)
            node = parent_f[node]
        path_forward.reverse()  

        path_backward = []
        node = parent_b[meeting_node]
        while node is not None:
            path_backward.append(node)
            node = parent_b[node]

        full_path = path_forward + path_backward
        return full_path

    def set_robot_path(self, robot):
        path = self.bidirectional_search(robot.current_pos, robot.goal_pos)

        if path is None or len(path) < 2:
            print(f"Robot {robot.id}: No path found from {robot.current_pos} to {robot.goal_pos}.")
        else:
            robot.path = path

    def update_robot_no_conflict(self, intended_moves):
        for pos, robot_list in intended_moves.items():

            if len(robot_list) == 1:
                robot = robot_list[0]
                robot.current_pos = pos
                robot.path_taken.append(pos)

                if len(robot.path) > 0:
                    robot.path.pop(0)
                print(f"Robot {robot.id} moves to {pos}.")

    def handle_collision(self, intended_moves):
        for pos, robot_list in intended_moves.items():

            if len(robot_list) > 1:
                colliding_ids = [r.id for r in robot_list]
                print(f"Collision detected at {pos} among robots {colliding_ids} at time {self.time + 1}.")

                for robot in robot_list:
                    valid_moves = self.get_valid_moves(robot.current_pos, self.time)

                    if valid_moves:
                        random_move = random.choice(valid_moves)
                        print(f"Robot {robot.id} randomly changes direction from {robot.current_pos} to {random_move}.")
                        robot.current_pos = random_move
                        robot.path_taken.append(random_move)
                        new_path = self.bidirectional_search(robot.current_pos, robot.goal_pos)

                        if new_path is None:
                            robot.path = []
                        else:
                            robot.path = new_path

                    else:
                        print(f"Robot {robot.id} has no valid moves from {robot.current_pos} at time {self.time}.")

    def run(self):
        for robot in self.robots.values():
            self.set_robot_path(robot)

        while self.time < self.max_time:
            print(f"\n--- Time step: {self.time} ---")
            intended_moves = {}

            for robot in self.robots.values():
                if robot.at_goal:
                    continue

                if robot.current_pos == robot.goal_pos:
                    robot.at_goal = True
                    robot.total_time = self.time
                    continue

                if not robot.path or robot.path[0] != robot.current_pos:
                    new_path = self.bidirectional_search(robot.current_pos, robot.goal_pos)

                    if new_path is None or len(new_path) < 2:
                        valid_moves = self.get_valid_moves(robot.current_pos, self.time)

                        if valid_moves:
                            robot.path = [robot.current_pos, random.choice(valid_moves)]
                        else:
                            print(f"Robot {robot.id} is stuck at {robot.current_pos} at time {self.time}.")
                            continue
                    else:
                        robot.path = new_path

                next_move = robot.path[1] if len(robot.path) >= 2 else robot.current_pos
                intended_moves.setdefault(next_move, []).append(robot)

            self.update_robot_no_conflict(intended_moves)
            self.handle_collision(intended_moves)

            self.time += 1

            if all(robot.at_goal or (robot.current_pos == robot.goal_pos) for robot in self.robots.values()):
                for robot in self.robots.values():
                    if not robot.at_goal and robot.current_pos == robot.goal_pos:
                        robot.at_goal = True
                        robot.total_time = self.time
                break

        print("\n=== Simulation Ended ===")
        for robot in self.robots.values():
            print(f"\nRobot {robot.id}:")
            print(f"  Start: {robot.start_pos}")
            print(f"  Goal: {robot.goal_pos}")
            print(f"  Total Time to reach goal: {robot.total_time if robot.total_time else 'Not reached'}")
            print(f"  Path taken: {robot.path_taken}")


def main():
    print("Select a dataset to use:")
    print("0. Dataset 0")
    print("1. Dataset 1")
    print("2. Dataset 2")
    print("3. Dataset 3")
    print("4. Dataset 4")
    
    choice = input("Enter the number corresponding to the dataset: ")
    
    if choice not in ['0', '1', '2', '3', '4']:
        print("Invalid choice. Defaulting to Dataset 0.")
        choice = '0'
    
    agent_file = f"Input_Files/Data/Agent{choice}.txt"
    robot_file = f"Input_Files/Data/Robots{choice}.txt"
    grid_file = f"Input_Files/Data/data{choice}.txt"
    
    parsed = Parsing(agent_file, robot_file, grid_file)
    dynamic_agents = parsed.agents_data
    robots = parsed.robots_data
    grid = parsed.grid_data
    
    world = World(grid, dynamic_agents, 0)
    simulation = Simulate(robots, world)
    simulation.run()

main()


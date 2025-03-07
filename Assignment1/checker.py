from typing import Dict, List, Optional, Tuple

from z3 import And, Or, Bool, CheckSatResult, If, Int, Not, Solver


def init_environment(timesteps: int, grid_size: Optional[int] = 10) -> Tuple[
    Solver,
    Dict[str, List[Int]],
    Dict[int, Dict[str, List[Int]]],
    Dict[int, Dict[str, List[Int]]],
    Dict[str, List[Bool]],
]:
    """
    Initializes the Z3 variables and constraints for the environment.

    Args:
        timesteps (int): The number of timesteps in the planning horizon.
        grid_size (int, optional): The size of the grid. Defaults to 10.

    Returns:
        Tuple[
            Solver,
            Dict[str, List[Int]],
            Dict[int, Dict[str, List[Int]]],
            Dict[int, Dict[str, List[Int]]],
            Dict[str, List[Bool]],
        ]:
            A tuple containing the following:
                Solver: The Z3 solver.
                Dict[str, List[Int]]: A dictionary containing the agent's position (row, column) at each timestep.
                Dict[int, Dict[str, List[Int]]]: A dictionary containing the targets' position (row, column) at each timestep.
                Dict[int, Dict[str, List[Int]]]: A dictionary containing the distances between each target and the agent at each timestep.
                Dict[str, List[Bool]]: A dictionary containing the agent's direction of movement at each timestep (0 for 'north', 1 for 'east', 2 for 'south', 3 for 'west').
    """
    # Dictionary to store the agent position (row, column) at each timestep
    agent_position: Dict[str, List[Int]] = {"row": [], "column": []}
    for axis in ["row", "column"]:
        agent_position[axis] = [Int("a_{}{}".format(axis, t)) for t in range(timesteps)]

    # Dictionary to store the targets' position (row, column) at each timestep
    target_positions: Dict[int, Dict[str, List[Int]]] = {
        0: {"row": [], "column": []},
        1: {"row": [], "column": []},
        2: {"row": [], "column": []},
    }
    for target in target_positions:
        for axis in ["row", "column"]:
            target_positions[target][axis] = [
                Int("t_{}{}{}".format(target, axis, t)) for t in range(timesteps)
            ]

    # Dictionary to store each of the target distances from the agent at each timestep
    targets_distance: Dict[int, Dict[str, List[Int]]] = {
        0: {"row": [], "column": []},
        1: {"row": [], "column": []},
        2: {"row": [], "column": []},
    }
    for target in targets_distance:
        for axis in ["row", "column"]:
            targets_distance[target][axis] = [
                Int("d_{}{}{}".format(target, axis, t)) for t in range(timesteps)
            ]

    # Dictionary to store each action at each timestep (boolean)
    agent_direction: Dict[str, List[Bool]] = {
        "north": [Bool("north_{}".format(t)) for t in range(timesteps)],
        "east": [Bool("east_{}".format(t)) for t in range(timesteps)],
        "south": [Bool("south_{}".format(t)) for t in range(timesteps)],
        "west": [Bool("west_{}".format(t)) for t in range(timesteps)],
    }

    # Initialize the solver
    solver = Solver()

    # Constraints to ensure that the agent and all the targets are within the grid at each timestep
    for t in range(timesteps):
        for axis in ["row", "column"]:
            for i in range(3):
                solver.add(target_positions[i][axis][t] < grid_size)
                solver.add(target_positions[i][axis][t] >= 0)

            solver.add(agent_position[axis][t] < grid_size)
            solver.add(agent_position[axis][t] >= 0)

    # Constraint to ensure that if one target is picked up, a new target appears at a different location
    for t in range(1, timesteps):
        for i in range(3):
            solver.add(
                If(  # if: the agent was at the target's position in the previous timestep
                    And(
                        target_positions[i]["row"][t - 1]
                        == agent_position["row"][t - 1],
                        target_positions[i]["column"][t - 1]
                        == agent_position["column"][t - 1],
                    ),
                    # then: either the row or column position of the target must have changed since the last timestep
                    Not(
                        And(
                            target_positions[i]["row"][t]
                            != target_positions[i]["row"][t - 1],
                            target_positions[i]["column"][t]
                            != target_positions[i]["column"][t - 1],
                        )
                    ),
                    # else: no further constraints
                    True,
                )
            )

    # Constraint to ensure that all the targets are in different locations at each timestep
    for t in range(timesteps):
        for i in range(3):
            for j in range(i):
                solver.add(
                    Not(
                        And(
                            target_positions[i]["row"][t]
                            == target_positions[j]["row"][t],
                            target_positions[i]["column"][t]
                            == target_positions[j]["column"][t],
                        )
                    )
                )

    # Constraint to ensure that distance between the agent and the targets is valid at each timestep
    for t in range(timesteps):
        for axis in ["row", "column"]:
            for i in range(3):
                solver.add(
                    targets_distance[i][axis][t]
                    == target_positions[i][axis][t] - agent_position[axis][t]
                )

    # Constraint to ensure that the agent can only move in one direction at each timestep
    for t in range(1, timesteps):
        solver.add(
            agent_position["row"][t]  # agent's current row position
            == agent_position["row"][
                t - 1
            ]  # is equal to the agent's previous row position
            + If(
                agent_direction["north"][t - 1],
                -1,  # if the agent moved north, then the agent's current row position is one less than the agent's previous row position
                If(
                    agent_direction["south"][t - 1], 1, False
                ),  # if the agent moved south, then the agent's current row position is one more than the agent's previous row position
            )
        )
        solver.add(
            agent_position["column"][t]  # agent's current column position
            == agent_position["column"][
                t - 1
            ]  # is equal to the agent's previous column position
            + If(
                agent_direction["east"][t - 1],
                1,  # if the agent moved west, then the agent's current column position is one more than the agent's previous column position
                If(
                    agent_direction["west"][t - 1], -1, False
                ),  # if the agent moved east, then the agent's current column position is one less than the agent's previous column position
            )
        )

    return (
        solver,
        agent_position,
        target_positions,
        targets_distance,
        agent_direction,
    )


def check_run(
    agent_position_list: List[List[int]],
    target_position_list: List[List[List[int]]],
    action_list: List[int],
    grid_size: Optional[int] = 10,
) -> CheckSatResult:
    """
    Check if the given sequence of actions leads the agent from the given initial positions to the target positions on a grid of
    the specified size without colliding with obstacles.

    Args:
        agent_position_list (List[List[int]]): A list of agent positions where each position is a list of two integers
        representing the coordinates (row and column) of the agent at that time step.
        target_position_list (List[List[List[int]]]): A list of target positions where each target position is a list of
        three lists of two integers representing the coordinates (row and column) of the three targets at that time step.
        action_list (List[int]): A list of integers representing the sequence of actions taken by the agent where 0 represents
        north, 1 represents east, 2 represents south, and 3 represents west.
        grid_size (Optional[int]): An optional integer representing the size of the square grid. Defaults to 10.

    Returns:
        CheckSatResult: An enumeration indicating whether the given sequence of actions leads to a valid solution or not.
    """
    # Initialize the environment
    (
        solver,
        agent_position,
        target_positions,
        _,
        agent_direction,
    ) = init_environment(len(action_list), grid_size)

    # Add the run's data as constraints to the solver
    for t in range(len(agent_position_list)):
        solver.add(agent_position["row"][t] == int(agent_position_list[t][0]))
        solver.add(agent_position["column"][t] == int(agent_position_list[t][1]))

        for i in range(3):
            solver.add(
                target_positions[i]["row"][t] == int(target_position_list[t][i][0])
            )
            solver.add(
                target_positions[i]["column"][t] == int(target_position_list[t][i][1])
            )

        solver.add(agent_direction["north"][t] == bool(action_list[t] == 0))
        solver.add(agent_direction["east"][t] == bool(action_list[t] == 1))
        solver.add(agent_direction["south"][t] == bool(action_list[t] == 2))
        solver.add(agent_direction["west"][t] == bool(action_list[t] == 3))

    return solver.check()


def find_loop(
    agent_position_list: List[List[int]],
    target_position_list: List[List[List[int]]],
    action_list: List[int],
    grid_size: Optional[int] = 10,
) -> CheckSatResult:
    """
    Checks if the path of the agent contains any loops of size 2, i.e.: the agent directly moving back to the previous square.

    Args:
        agent_position_list (List[List[int]]): A list of agent positions where each position is a list of two integers
        representing the coordinates (row and column) of the agent at that time step.
        target_position_list (List[List[List[int]]]): A list of target positions where each target position is a list of
        three lists of two integers representing the coordinates (row and column) of the three targets at that time step.
        action_list (List[int]): A list of integers representing the sequence of actions taken by the agent where 0 represents
        north, 1 represents east, 2 represents south, and 3 represents west.
        grid_size (Optional[int]): An optional integer representing the size of the square grid. Defaults to 10.
    Returns:
        CheckSatResult: An enumeration indicating whether the given sequence is free of loops (sat) or not (unsat).
    """
    (
        solver,
        agent_position,
        target_positions,
        _,
        agent_direction,
    ) = init_environment(len(action_list), grid_size)

    # Add the run's data as constraints to the solver
    for t in range(len(agent_position_list)):
        solver.add(agent_position["row"][t] == int(agent_position_list[t][0]))
        solver.add(agent_position["column"][t] == int(agent_position_list[t][1]))

        for i in range(3):
            solver.add(
                target_positions[i]["row"][t] == int(target_position_list[t][i][0])
            )
            solver.add(
                target_positions[i]["column"][t] == int(target_position_list[t][i][1])
            )

        solver.add(agent_direction["north"][t] == bool(action_list[t] == 0))
        solver.add(agent_direction["east"][t] == bool(action_list[t] == 1))
        solver.add(agent_direction["south"][t] == bool(action_list[t] == 2))
        solver.add(agent_direction["west"][t] == bool(action_list[t] == 3))


    for t in range(len(action_list) - 1):
        # To satisfy condition that agent does not do 2-step loops unless it has reached a target,
        # we check at each time step that the event that none of the targets has been reached and one of the
        # four possible 2-step loops has NOT happened. Exact formula below:
        #
        # For each time step:
        #   T_1 - agent has not reached first target
        #   T_2 - agent has not reached second target
        #   T_3 - agent has not reached third target
        #   L_NS - agent has gone north and immediately south
        #   L_SN - agent has gone south and immediately north
        #   L_WE - agent has gone west and immediately east
        #   L_EW - agent has gone east and immediately west
        #   condition: ~(T_1 ^ T_2 ^ T_3 ^ (L_NS v L_SN v L_WE v L_EW))
        solver.add(
            Not(
                And(
                    Or( # T_1 - agent has not reached first target
                        agent_position["row"][t + 1] != target_positions[0]["row"][t],
                        agent_position["column"][t + 1]
                        != target_positions[0]["column"][t],
                    ),
                    Or( # T_2 - agent has not reached second target
                        agent_position["row"][t + 1] != target_positions[1]["row"][t],
                        agent_position["column"][t + 1]
                        != target_positions[1]["column"][t],
                    ),
                    Or( # T_3 - agent has not reached second target
                        agent_position["row"][t + 1] != target_positions[2]["row"][t],
                        agent_position["column"][t + 1]
                        != target_positions[2]["column"][t],
                    ),
                    Or(
                        And( # L_NS - agent has gone north and immediately south
                            agent_direction["north"][t],
                            agent_direction["south"][t + 1],
                        ),
                        And( # L_SN - agent has gone south and immediately north
                            agent_direction["south"][t],
                            agent_direction["north"][t + 1],
                        ),
                        And( # L_WE - agent has gone west and immediately east
                            agent_direction["west"][t],
                            agent_direction["east"][t + 1],
                        ),
                        And( # L_EW - agent has gone east and immediately west
                            agent_direction["east"][t],
                            agent_direction["west"][t + 1],
                        ),
                    ),
                ),
            )
        )

    return solver.check()


def find_efficient_path(
    agent_position_list: List[List[int]],
    target_position_list: List[List[List[int]]],
    action_list: List[int],
    grid_size: Optional[int] = 10,
) -> CheckSatResult:
    """
    Checks if the path of the agent took the most efficient path to a target.

    Args:
        agent_position_list (List[List[int]]): A list of agent positions where each position is a list of two integers
        representing the coordinates (row and column) of the agent at that time step.
        target_position_list (List[List[List[int]]]): A list of target positions where each target position is a list of
        three lists of two integers representing the coordinates (row and column) of the three targets at that time step.
        action_list (List[int]): A list of integers representing the sequence of actions taken by the agent where 0 represents
        north, 1 represents east, 2 represents south, and 3 represents west.
        grid_size (Optional[int]): An optional integer representing the size of the square grid. Defaults to 10.
    Returns:
        CheckSatResult: An enumeration indicating whether the given sequence the shortest path (sat) or not (unsat).
    """
    (
        solver,
        agent_position,
        target_positions,
        _,
        agent_direction,
    ) = init_environment(len(action_list), grid_size)

    # Add the run's data as constraints to the solver
    for t in range(len(agent_position_list)):
        solver.add(agent_position["row"][t] == int(agent_position_list[t][0]))
        solver.add(agent_position["column"][t] == int(agent_position_list[t][1]))

        for i in range(3):
            solver.add(
                target_positions[i]["row"][t] == int(target_position_list[t][i][0])
            )
            solver.add(
                target_positions[i]["column"][t] == int(target_position_list[t][i][1])
            )

        solver.add(agent_direction["north"][t] == bool(action_list[t] == 0))
        solver.add(agent_direction["east"][t] == bool(action_list[t] == 1))
        solver.add(agent_direction["south"][t] == bool(action_list[t] == 2))
        solver.add(agent_direction["west"][t] == bool(action_list[t] == 3))

    # I find the list of targets picked by the agent by checking at each time step
    # if any of the targets' coordinates have changed
    chosen_targets = []
    for t in range(len(action_list) - 1):
        for i in range(3):
            if (
                target_position_list[t][i][0] != target_position_list[t + 1][i][0]
                or target_position_list[t][i][1] != target_position_list[t + 1][i][1]
            ):
                chosen_targets.append((i, t + 1))

    current_target_it = 0
    current_target, arr_time = chosen_targets[current_target_it]
    for t in range(len(action_list)):
        # At each time step, I update the currently chosen target if necessery by comparing
        # the time with the arrival time at given target
        if t == arr_time:
            current_target_it += 1
            if current_target_it == len(chosen_targets):
                break
            current_target, arr_time = chosen_targets[current_target_it]
        # To ensure that the path to the chosen target is optimal, it's enough to check
        # that the agent is not moving away from the chosen target in any of the two dimensions

        # If the currently chosen target is not below the agent, the agent is not allowed
        # to move south (NOTE: important that it's NOT BELOW instead of ABOVE; this ensures that
        # if the target is directly north, south, east or west of the agent, it will only be allowed
        # to move in one correct dimension)
        solver.add(
            If(
                target_positions[current_target]["row"][t] <= agent_position["row"][t],
                Not(agent_direction["south"][t]),
                True,
            )
        )
        # If the currently chosen target is not above the agent, the agent is not allowed
        # to move north
        solver.add(
            If(
                target_positions[current_target]["row"][t] >= agent_position["row"][t],
                Not(agent_direction["north"][t]),
                True,
            )
        )
        # If the currently chosen target is not to the right of the agent, the agent is not allowed
        # to move east
        solver.add(
            If(
                target_positions[current_target]["column"][t]
                <= agent_position["column"][t],
                Not(agent_direction["east"][t]),
                True,
            )
        )
        # If the currently chosen target is not to the left of the agent, the agent is not allowed
        # to move west
        solver.add(
            If(
                target_positions[current_target]["column"][t]
                >= agent_position["column"][t],
                Not(agent_direction["west"][t]),
                True,
            )
        )

    return solver.check()


def closest_target(
    agent_position_list: List[List[int]],
    target_position_list: List[List[List[int]]],
    action_list: List[int],
    grid_size: Optional[int] = 10,
) -> CheckSatResult:
    """
    Checks if the path of the agent was to the closest possible target

    Args:
        agent_position_list (List[List[int]]): A list of agent positions where each position is a list of two integers
        representing the coordinates (row and column) of the agent at that time step.
        target_position_list (List[List[List[int]]]): A list of target positions where each target position is a list of
        three lists of two integers representing the coordinates (row and column) of the three targets at that time step.
        action_list (List[int]): A list of integers representing the sequence of actions taken by the agent where 0 represents
        north, 1 represents east, 2 represents south, and 3 represents west.
        grid_size (Optional[int]): An optional integer representing the size of the square grid. Defaults to 10.
    Returns:
        CheckSatResult: An enumeration indicating whether the given sequence to the closest target (sat) or not (unsat).
    """
    # TODO: Implement this function as specified. Note that you may not need all arguments made available to you.
    (
        solver,
        agent_position,
        target_positions,
        targets_distance,
        agent_direction,
    ) = init_environment(len(action_list), grid_size)
    for t in range(len(agent_position_list)):
        solver.add(agent_position["row"][t] == int(agent_position_list[t][0]))
        solver.add(agent_position["column"][t] == int(agent_position_list[t][1]))

        for i in range(3):
            solver.add(
                target_positions[i]["row"][t] == int(target_position_list[t][i][0])
            )
            solver.add(
                target_positions[i]["column"][t] == int(target_position_list[t][i][1])
            )

        solver.add(agent_direction["north"][t] == bool(action_list[t] == 0))
        solver.add(agent_direction["east"][t] == bool(action_list[t] == 1))
        solver.add(agent_direction["south"][t] == bool(action_list[t] == 2))
        solver.add(agent_direction["west"][t] == bool(action_list[t] == 3))

    # I find the list of targets picked by the agent by checking at each time step
    # if any of the targets' coordinates have changed. I also save times of choosing the next target
    chosen_targets = []
    choosing_times = [0]
    for t in range(len(action_list) - 1):
        for i in range(3):
            if (
                target_position_list[t][i][0] != target_position_list[t + 1][i][0]
                or target_position_list[t][i][1] != target_position_list[t + 1][i][1]
            ):
                chosen_targets.append(i)
                choosing_times.append(t + 1)
    choosing_times.pop()

    # At each time step when a new target is chosen, I check that the chosen target is the closest
    # (or one of the closest) to the agent at the time of choosing
    for i in range(len(chosen_targets)):
        choosing_time = choosing_times[i]
        chosen_target = chosen_targets[i]
        for target in range(3):
            if target == chosen_target:
                continue
            solver.add(
                targets_distance[chosen_target]["row"][choosing_time]
                + targets_distance[chosen_target]["column"][choosing_time] <=
                targets_distance[target]["row"][choosing_time] + targets_distance[target]["column"][choosing_time]
            )
    return solver.check()

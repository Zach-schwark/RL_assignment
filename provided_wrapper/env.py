import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete, Box

import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward
from grid2op.gym_compat import  BoxGymObsSpace, DiscreteActSpace, BoxGymActSpace, MultiDiscreteActSpace

from lightsim2grid import LightSimBackend


# Gymnasium environment wrapper around Grid2Op environment
class Gym2OpEnv(gym.Env):
    def __init__(
            self,
            baseline: bool,
            first_iteraion: bool,
            second_iteraion: bool,
    ):
        super().__init__()
        
        self.baseline = baseline
        self.first_iteration = first_iteraion
        self.second_iteration = second_iteraion

        self._backend = LightSimBackend()
        self._env_name = "l2rpn_case14_sandbox"  # DO NOT CHANGE

        action_class = PlayableAction
        observation_class = CompleteObservation
        reward_class = CombinedScaledReward  # Setup further below

        # DO NOT CHANGE Parameters
        # See https://grid2op.readthedocs.io/en/latest/parameters.html
        p = Parameters()
        p.MAX_SUB_CHANGED = 4  # Up to 4 substations can be reconfigured each timestep
        p.MAX_LINE_STATUS_CHANGED = 4  # Up to 4 powerline statuses can be changed each timestep

        # Make grid2op env
        self._g2op_env = grid2op.make(
            self._env_name, backend=self._backend, test=False,
            action_class=action_class, observation_class=observation_class,
            reward_class=reward_class, param=p
        )

        ##########
        # REWARD #
        ##########
        # NOTE: This reward should not be modified when evaluating RL agent
        # See https://grid2op.readthedocs.io/en/latest/reward.html
        cr = self._g2op_env.get_reward_instance()
        cr.addReward("N1", N1Reward(), 1.0)
        cr.addReward("L2RPN", L2RPNReward(), 1.0)
        # reward = N1 + L2RPN
        cr.initialize(self._g2op_env)
        ##########

        self._gym_env = gym_compat.GymEnv(self._g2op_env)

        self.setup_observations()
        self.setup_actions()

        self.observation_space = self._gym_env.observation_space
        self.action_space = self._gym_env.action_space
        
        
    # The information i used to get the code for the below 2 functions is fromm the getting started from the Grid2Op github.
    # specifcally theis link was useful at helping change the observation and action spaces to use with gymnasium : https://github.com/rte-france/Grid2Op/blob/c71a2dfb824dae7115394266e02cc673c8633a0e/getting_started/11_IntegrationWithExistingRLFrameworks.ipynb
    # these links also help explain the observation and action space: 
        # https://github.com/rte-france/Grid2Op/blob/c71a2dfb824dae7115394266e02cc673c8633a0e/getting_started/02_Observation.ipynb
        # https://github.com/rte-france/Grid2Op/blob/c71a2dfb824dae7115394266e02cc673c8633a0e/getting_started/03_Action.ipynb

    def setup_observations(self):
        
        attr_to_keep_3 = ['a_ex', 'a_or', 'active_alert', 'actual_dispatch', 'alert_duration','attention_budget', 'current_step','curtailment', 'curtailment_limit', 'curtailment_limit_effective', 'curtailment_limit_mw', 'curtailment_mw','day','day_of_week', 'delta_time', 'duration_next_maintenance', 'gen_margin_down','gen_margin_up', 'gen_p', 'gen_p_before_curtail', 'gen_q', 'gen_theta', 'gen_v','hour_of_day', 'last_alarm', 'line_status', 'load_p', 'load_q','load_theta', 'load_v', 'max_step', 'minute_of_hour', 'month', 'p_ex', 'p_or','prod_p', 'prod_q', 'prod_v', 'q_ex', 'q_or', 'rho', 'storage_charge','storage_power', 'storage_power_target', 'storage_theta', 'target_dispatch','thermal_limit', 'theta_ex', 'theta_or', 'time_before_cooldown_line','time_before_cooldown_sub', 'time_next_maintenance','timestep_overflow', 'topo_vect','v_ex', 'v_or', 'year']#so far the best
        
        attr_to_keep_3_remove_storage = ['a_ex', 'a_or', 'active_alert', 'actual_dispatch', 'alert_duration','attention_budget', 'current_step','curtailment', 'curtailment_limit', 'curtailment_limit_effective', 'curtailment_limit_mw', 'curtailment_mw','day','day_of_week', 'delta_time', 'duration_next_maintenance', 'gen_margin_down','gen_margin_up', 'gen_p', 'gen_p_before_curtail', 'gen_q', 'gen_theta', 'gen_v','hour_of_day', 'last_alarm', 'line_status', 'load_p', 'load_q','load_theta', 'load_v', 'max_step', 'minute_of_hour', 'month', 'p_ex', 'p_or','prod_p', 'prod_q', 'prod_v', 'q_ex', 'q_or', 'rho', 'target_dispatch','thermal_limit', 'theta_ex', 'theta_or', 'time_before_cooldown_line','time_before_cooldown_sub', 'time_next_maintenance','timestep_overflow', 'topo_vect','v_ex', 'v_or', 'year']#so far the best

        #attr_to_keep_4 = ['a_ex', 'a_or',  'actual_dispatch', 'attention_budget', 'current_step','curtailment', 'curtailment_limit', 'curtailment_limit_effective', 'curtailment_limit_mw', 'curtailment_mw','day','day_of_week', 'delta_time', 'duration_next_maintenance', 'gen_margin_down','gen_margin_up', 'gen_p', 'gen_p_before_curtail', 'gen_q', 'gen_theta', 'gen_v','hour_of_day', 'line_status', 'load_p', 'load_q','load_theta', 'load_v', 'max_step', 'minute_of_hour', 'month', 'p_ex', 'p_or','prod_p', 'prod_q', 'prod_v', 'q_ex', 'q_or', 'rho', 'storage_charge','storage_power', 'storage_power_target', 'storage_theta', 'target_dispatch','thermal_limit', 'theta_ex', 'theta_or', 'time_before_cooldown_line','time_before_cooldown_sub', 'time_next_maintenance','timestep_overflow', 'topo_vect','v_ex', 'v_or', 'year'] #this one is bad so active alert and alert duration is important

        #attr_to_keep_5 = ['a_ex', 'a_or', 'active_alert', 'actual_dispatch', 'alert_duration','attention_budget', 'current_step','curtailment', 'curtailment_limit', 'curtailment_limit_effective', 'curtailment_limit_mw', 'curtailment_mw','day','day_of_week', 'delta_time', 'duration_next_maintenance', 'gen_margin_down','gen_margin_up', 'gen_p', 'gen_p_before_curtail', 'gen_q', 'gen_theta', 'gen_v','hour_of_day', 'last_alarm', 'line_status', 'load_p', 'load_q','load_theta', 'load_v', 'max_step', 'minute_of_hour', 'month', 'p_ex', 'p_or','prod_p', 'prod_q', 'prod_v', 'q_ex', 'q_or', 'rho', 'storage_power', 'storage_power_target', 'storage_theta', 'target_dispatch','thermal_limit', 'theta_ex', 'theta_or', 'time_before_cooldown_line','time_before_cooldown_sub', 'time_next_maintenance','timestep_overflow', 'topo_vect','v_ex', 'v_or', 'year'] #removed storage charged - does very bad

        #attr_to_keep_6 = ['a_ex', 'a_or', 'active_alert', 'actual_dispatch', 'alert_duration','attention_budget', 'current_step','curtailment', 'curtailment_limit', 'curtailment_limit_effective', 'curtailment_limit_mw', 'curtailment_mw','day','day_of_week', 'delta_time', 'duration_next_maintenance', 'gen_margin_down','gen_margin_up', 'gen_p', 'gen_p_before_curtail', 'gen_q', 'gen_theta', 'gen_v','hour_of_day', 'last_alarm', 'line_status', 'load_p', 'load_q','load_theta', 'load_v', 'max_step', 'minute_of_hour', 'month', 'p_ex', 'p_or','prod_p', 'prod_q', 'prod_v', 'q_ex', 'q_or', 'rho', 'storage_charge','storage_power', 'storage_power_target', 'storage_theta', 'target_dispatch','thermal_limit', 'theta_ex', 'theta_or', 'time_before_cooldown_line', 'time_next_maintenance','timestep_overflow', 'topo_vect','v_ex', 'v_or', 'year']

        attr_to_keep = ["rho", "gen_p", "load_p", "topo_vect", "actual_dispatch"]

        if self.baseline == True:
            #self._gym_env.action_space.close()
            self._gym_env.observation_space = self._gym_env.observation_space
        elif self.first_iteration == True:
            #self._gym_env.action_space.close()
            self._gym_env.observation_space = self._gym_env.observation_space.keep_only_attr(attr_to_keep_3_remove_storage)
        elif self.second_iteration == True:
            #self._gym_env.action_space.close()
            self._gym_env.observation_space = self._gym_env.observation_space.keep_only_attr(attr_to_keep_3_remove_storage)
        
        


    def setup_actions(self):
        
        
        if self.baseline == True:
            #self._gym_env.action_space.close()
            self._gym_env.action_space = DiscreteActSpace(self._g2op_env.action_space)
            #self._gym_env.action_space = BoxGymActSpace(self._g2op_env.action_space)
        elif self.first_iteration == True:
            #self._gym_env.action_space.close()
            attr_to_keep = ["set_bus", "set_line_status"]
            self._gym_env.action_space = DiscreteActSpace(self._g2op_env.action_space, attr_to_keep=attr_to_keep)
            #attr_to_keep = ["redispatch", "set_storage", "curtail"]
            #self._gym_env.action_space = BoxGymActSpace(self._g2op_env.action_space, attr_to_keep=attr_to_keep)
        elif self.second_iteration == True:
           # self._gym_env.action_space.close()
            attr_to_keep = ["set_bus", "set_line_status"]
            self._gym_env.action_space = DiscreteActSpace(self._g2op_env.action_space, attr_to_keep=attr_to_keep)
            #attr_to_keep = ["redispatch", "set_storage", "curtail"]
            #self._gym_env.action_space = BoxGymActSpace(self._g2op_env.action_space, attr_to_keep=attr_to_keep)
        

    def reset(self, seed=None):
        return self._gym_env.reset(seed=seed, options=None)

    def step(self, action):
        return self._gym_env.step(action)

    def render(self):
        # TODO: Modify for your own required usage
        return self._gym_env.render()


def main():
    # Random agent interacting in environment #

    max_steps = 100

    env = Gym2OpEnv(baseline=True, first_iteraion=False,second_iteraion=False)

    print("#####################")
    print("# OBSERVATION SPACE #")
    print("#####################")
    print(env.observation_space)
    print("#####################\n")

    print("#####################")
    print("#   ACTION SPACE    #")
    print("#####################")
    print(env.action_space)
    print("#####################\n\n")

    curr_step = 0
    curr_return = 0

    is_done = False
    obs, info = env.reset()
    print(f"step = {curr_step} (reset):")
    print(f"\t obs = {obs}")
    print(f"\t info = {info}\n\n")

    while not is_done and curr_step < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(0)

        curr_step += 1
        curr_return += reward
        is_done = terminated or truncated

        #print(f"step = {curr_step}: ")
        #print(f"\t obs = {obs}")
        #print(f"\t reward = {reward}")
        #print(f"\t terminated = {terminated}")
        #print(f"\t truncated = {truncated}")
        #print(f"\t info = {info}")

        # Some actions are invalid (see: https://grid2op.readthedocs.io/en/latest/action.html#illegal-vs-ambiguous)
        # Invalid actions are replaced with 'do nothing' action
        is_action_valid = not (info["is_illegal"] or info["is_ambiguous"])
        #print(f"\t is action valid = {is_action_valid}")
        if not is_action_valid:
            print(f"\t\t reason = {info['exception']}")
        #print("\n")

    print("###########")
    print("# SUMMARY #")
    print("###########")
    print(f"return = {curr_return}")
    print(f"total steps = {curr_step}")
    print("###########")


if __name__ == "__main__":
    main()

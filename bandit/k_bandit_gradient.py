import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
# parameters to set #######################
n_bandit = 10
variance_reward_init = 1
variance_reward = 1
mean_reward_init = 4
n_step = 1000
n_sim = 2000
print_every = 10000
print_every_sim = 10
stationary_mode = True
variance_nonstationary = 0  # 0.01
mean_nonstationary = 0
parameters = [{'Q':0, 'alpha':0.1, 'reward_baseline_on': 1}, {'Q':0, 'alpha':0.1,'reward_baseline_on': 0},
              {'Q':0, 'alpha':0.4, 'reward_baseline_on': 1}, {'Q':0, 'alpha':0.4,'reward_baseline_on': 0}]
# parameters to set #######################

# make sure that variance_nonstationary and mean_nonstationary are 0 for stationary case
variance_nonstationary = 0 if stationary_mode else variance_nonstationary
mean_nonstationary = 0 if stationary_mode else variance_nonstationary

def sotfmax(preference_vals_in):
    preference_vals = preference_vals_in.flatten()
    preference_vals = preference_vals-np.max(preference_vals)
    preference_probs = np.zeros_like(preference_vals)
    numer = np.exp(preference_vals)
    preference_probs = numer/(np.sum(numer))
    return preference_probs


fig ,ax = plt.subplots()
fig2, ax2 = plt.subplots()
for cur_params, cur_params_idx in zip(parameters, range(len(parameters))):
    alpha = cur_params['alpha']
    action_value_init = cur_params['Q']
    reward_baseline_on = cur_params['reward_baseline_on']
    average_reward = np.zeros((n_sim, n_step))
    actions_counts = np.zeros((n_sim, n_bandit), dtype=np.int32)
    action_counts_ratio = np.zeros((n_sim, n_step))
    action_counts_true = np.zeros((n_sim, n_step))
    gt_actions = np.zeros((n_sim, 1), dtype=np.int32)
    for sim_idx in range(n_sim):
        reward_means = np.random.randn(1, n_bandit)*variance_reward_init+mean_reward_init
        gt_actions[sim_idx, 0] = np.argmax(reward_means)
        # action_values_hist = np.matlib.repmat(reward_means, n_step+1, 1)
        action_values_hist = np.zeros((n_step+1, n_bandit))
        action_values_init = np.ones((1, n_bandit))*action_value_init
        action_values_hist[0, :] = action_values_init
        cur_sim_average_reward = action_value_init
        preference_values = np.zeros((n_step+1, n_bandit))
        for i in range(n_step):
            preference_probs = sotfmax(preference_values[i, :])
            action = np.random.choice(np.arange(n_bandit), 1, p=preference_probs)
            not_action = np.delete(np.arange(n_bandit), action)
            cur_reward = np.random.randn()*variance_reward+reward_means[0, action]
            reward_means+=np.random.randn(1, n_bandit)*variance_nonstationary+mean_nonstationary
            average_reward[sim_idx, i] = cur_reward
            actions_counts[sim_idx, action]+=1
            action_counts_ratio[sim_idx, i] = actions_counts[sim_idx, gt_actions[sim_idx, 0]]/(i+1)
            cur_sim_average_reward = cur_sim_average_reward+1.0/(i+1)*(cur_reward-cur_sim_average_reward)
            if not reward_baseline_on:
                cur_sim_average_reward = 0
            update_value = cur_reward-cur_sim_average_reward
            preference_values[i+1, action] = preference_values[i, action]+alpha*update_value*(1.0-preference_probs[action])
            preference_values[i+1, not_action] = preference_values[i, not_action]-alpha*update_value*preference_probs[not_action]
            
            if action == gt_actions[sim_idx, 0]:
                action_counts_true[sim_idx, i] = 1
            action_values_hist[i+1, :] = action_values_hist[i, :]
            if stationary_mode:
                alpha_coeff = 1.0/(actions_counts[sim_idx, action])
            else:
                alpha_coeff = alpha
            action_values_hist[i+1, action] = action_values_hist[i, action]+alpha_coeff*(cur_reward-action_values_hist[i, action])
            if (i+1) % print_every == 0:
                print('step progress: {}/{}, {}/{}'.format(i+1, n_step, sim_idx+1, n_sim))
        if (sim_idx+1)%print_every_sim == 0:
            print('sim progress: {}/{}, param progress: {}/{}, Q: {}, alpha: {}'.format(sim_idx+1, n_sim, 
                    cur_params_idx+1, len(parameters), action_value_init, alpha))
    label_txt = 'Q1: {}, alpha: {}, reward baseline: {}'.format(action_value_init, alpha, reward_baseline_on)
    ax.plot(np.mean(average_reward, axis=0), label=label_txt)
    ax.set_xlabel('Step')
    ax.set_ylabel('Average Return')
    ax2.plot(np.sum(action_counts_true, axis=0)/n_sim*100, label=label_txt)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Optimal Action (%)')
    print('param progress: {}/{}'.format(cur_params_idx+1, len(parameters)))
ax.legend()
ax2.legend()
plt.show()

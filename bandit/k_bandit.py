import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
# parameters to set #######################
n_bandit = 10
variance_reward_init = 1
variance_reward = 1
mean_reward_init = 0
n_step = 1000
n_sim = 1000
print_every = 20000
print_every_sim = 10
stationary_mode = True
alpha = None if stationary_mode else 0.1
variance_nonstationary = 0  # 0.01
mean_nonstationary = 0
parameters = [
            {'Q':0, 'eps':0, 'ucb':1, 'c':1/64}, {'Q':0, 'eps':0, 'ucb':1, 'c': 1/32},
            {'Q':0, 'eps':0, 'ucb':1, 'c':1/16}, {'Q':0, 'eps':0, 'ucb':1, 'c': 1/8},
            {'Q':0, 'eps':0, 'ucb':1, 'c':1/4}, {'Q':0, 'eps':0, 'ucb':1, 'c': 1/2},
            {'Q':0, 'eps':0, 'ucb':1, 'c':1}, {'Q':0, 'eps':0, 'ucb':1, 'c': 2},
            {'Q':0, 'eps':0, 'ucb':1, 'c':4},  {'Q':0, 'eps':0, 'ucb':1, 'c': 8},
            ]
# parameters to set #######################
avg_scores = np.zeros((1, len(parameters)))
step_begin = 0
step_end = n_step
assert step_begin<=step_end<=n_step, "step begin and step end parameters are wrong check them again"
fig ,ax = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
for cur_params, cur_params_idx in zip(parameters, range(len(parameters))):
    eps = cur_params['eps']
    action_value_init = cur_params['Q']
    ucb_on = cur_params['ucb']
    degree_of_eploration = cur_params['c']
    assert degree_of_eploration == None or degree_of_eploration>0, 'degre_of_exploration value is not valid'
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
        for i in range(n_step):
            if np.random.rand()>eps:
                if ucb_on:
                    action = np.argmax(action_values_hist[i,:]+degree_of_eploration*np.sqrt(np.log(i+1)/(actions_counts[sim_idx,:]+1e-6)))
                else:
                    action = np.argmax(action_values_hist[i,:])
            else:
                action = np.random.randint(0, n_bandit)
            cur_reward = np.random.randn()*variance_reward+reward_means[0, action]
            reward_means+=np.random.randn(1, n_bandit)*variance_nonstationary+mean_nonstationary
            average_reward[sim_idx, i] = cur_reward
            actions_counts[sim_idx, action]+=1
            action_counts_ratio[sim_idx, i] = actions_counts[sim_idx, gt_actions[sim_idx, 0]]/(i+1)
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
            print('sim progress: {}/{}, param progress: {}/{}, Q: {}, eps: {}, UCB: {}, c: {}'.format(sim_idx+1, n_sim, 
                    cur_params_idx+1, len(parameters), action_value_init, eps, ucb_on, degree_of_eploration ))
    label_txt = 'Q1: {}, eps: {}'.format(action_value_init, eps)
    if ucb_on:
        label_txt+=', UCB, c: {}'.format(degree_of_eploration)
    ax.plot(np.mean(average_reward, axis=0), label=label_txt)
    # ax2.plot(np.sum(actions_counts[np.arange(n_sim), gt_actions.flatten().astype(int)])/(n_sim*n_step), label='eps: '+str(eps))
    # ax2.plot(np.mean(action_counts_ratio, axis=0)*100, label='Q1: {}, eps: {}'.format(action_value_init, eps))
    ax2.plot(np.sum(action_counts_true, axis=0)/n_sim*100, label=label_txt)
    print('param progress: {}/{}'.format(cur_params_idx+1, len(parameters)))
    avg_scores[0, cur_params_idx] = np.mean(np.mean(average_reward[:,step_begin:step_end:], axis=0))
ax.legend()
ax2.legend()
x_axis = np.power(2.0, np.arange(-6, 4, 1))
ax3.plot(x_axis, avg_scores.flatten())
ax3.set_xscale('log')
ax3.set_xticks(ticks=x_axis.tolist())
ax3.set_xticklabels(labels=['1/64', '1/32', '1/16', '1/8', '1/4', '1/2', '1', '2', '4', '8'])
plt.show()
print('avg scores: ', avg_scores)

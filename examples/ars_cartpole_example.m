envName = 'CartPole-Continuous';
env = rlPredefinedEnv(envName);

nDelta = 32;
nTop = 32;
stepSize = .3;
deltaStd = .05;
nEpochs = 1000;

agent = ARSAgent(env, stepSize, deltaStd, nDelta, nTop, useBias=true);
rewards = agent.learn(nEpochs, verbose=1);

% Optionally comment out this line, to see your environment in action
%env.plot()  

[R,X] = doArsRollout(agent.policy, agent.env);

plot(X);
title(strcat(envName,' Rollout'))
xlabel('Time Step')
ylabel('State')

figure()
plot(rewards);
title(strcat(envName,' Reward Curve'))
xlabel('Iteration')
ylabel('Avg Rollout Reward')
envName = 'DoubleIntegrator-Continuous';
env = rlPredefinedEnv(envName);

nDelta = 32;
nTop = 32;
stepSize = .2;
deltaStd = .005;
nEpochs = 500;

% We can add a reward shift to each step of the environment. Generally we
% want rewards to be positive, or else the agent will find a local minima 
% where it terminates the episode early. 

agent = ARSAgent(env, stepSize, deltaStd, nDelta, nTop, useBias=false, rewardShift=10); 
rewards = agent.learn(nEpochs, verbose=1);

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
%% 

envName = 'CartPole-Continuous';
env = rlPredefinedEnv(envName);

nDelta = 32;
nTop = 32;
stepSize = .3;
deltaStd = .05;
nEpochs = 1000;

begin = tic;
agent = ARSAgent(env, stepSize, deltaStd, nDelta, nTop, useBias=true);

rewards = agent.learn(nEpochs, verbose=1);

fprintf("Avg Episodes Per Second: %f \n",  nEpochs*2*nDelta/toc(begin));


%env.plot()

[R,X] = doArsRollout(agent.policy, agent.env);
plot(X);
figure()
plot(rewards);
title(strcat(envName,' Reward Curve'))
xlabel('Iteration')
ylabel('Avg Rollout Reward')


%% 

envName = 'DoubleIntegrator-Continuous';
env = rlPredefinedEnv(envName);

nDelta = 32;
nTop = 32;
stepSize = .2;
deltaStd = .005;
nEpochs = 500;

begin = tic;
agent = ARSAgent(env, stepSize, deltaStd, nDelta, nTop, useBias=false, rewardShift=10);

rewards = agent.learn(nEpochs, verbose=1);

fprintf("Avg Episodes Per Second: %f \n",  nEpochs*2*nDelta/toc(begin));


[R,X] = doArsRollout(agent.policy, agent.env);
plot(X);
figure()
plot(rewards);
title(strcat(envName,' Reward Curve'))
xlabel('Iteration')
ylabel('Avg Rollout Reward')
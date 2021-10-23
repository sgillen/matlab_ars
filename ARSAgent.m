classdef ARSAgent < handle
    properties
        env % must be a valid RL env, see example_env.m
        stepSize(1,1) double
        deltaStd(1,1) double 
        nDelta(1,1) int64
        nTop(1,1) int64
        maxStepsPerEpisode(1,1)
        useBias(1,1) logical
        logInterval(1,1) int64 = 10;
        
        weights
        policy
        state_means 
        state_stds
        rewards
        
    end
    
    methods
        function obj = ARSAgent(env, stepSize, deltaStd, nDelta, nTop, optionalArgs)
            arguments
                env % % Must be a valid env, see example_env.m 
                stepSize(1,1) double {mustBePositive} = 0.02
                deltaStd(1,1) double {mustBePositive} = 0.03
                nDelta(1,1) int64 {mustBePositive} = 32
                nTop(1,1) int64 {mustBePositive} = nDelta
                optionalArgs.useBias(1,1) logical = false
                optionalArgs.maxStepsPerEpisode(1,1) int64 {mustBePositive} = 1000
                optionalArgs.logInterval(1,1) int64 {mustBePositive} = 10;
            end 
            obj.env = env;
            obj.stepSize = stepSize;
            obj.deltaStd = deltaStd;
            obj.nDelta = nDelta;
            obj.nTop = nTop;
            obj.maxStepsPerEpisode = optionalArgs.maxStepsPerEpisode;
            obj.useBias = optionalArgs.useBias;
            
            
            obj.weights = zeros(env.getObservationInfo.Dimension(1), env.getActionInfo.Dimension(1) + obj.useBias);
            obj.state_means = zeros(env.getObservationInfo.Dimension(1),1);
            obj.state_stds = ones(env.getObservationInfo.Dimension(1),1);

        end
        
        function rewards = learn(obj, nEpochs, optionalArgs)
           arguments
            obj;
            nEpochs(1,1) int64 {mustBePositive}
            optionalArgs.verbose(1,1) int64 {mustBeNonnegative} = 0;
           end
          
           verbose = optionalArgs.verbose;
           
           % Copy into local variables for performance
           env = obj.env;
           stepSize = obj.stepSize;
           deltaStd= obj.deltaStd;
           nDelta = obj.nDelta;
           nTop = obj.nTop;
           weights = obj.weights;
           useBias = obj.useBias;
           state_means = obj.state_means;
           state_stds = obj.state_stds;
           maxEnvSteps = obj.maxStepsPerEpisode;
           
           rewards = zeros(nEpochs,1);

           totalT = 0;

           var_thresh = 1e-6;

           for epoch = 1:nEpochs
                deltas = randn(size(weights,1), size(weights,2), nDelta)*deltaStd;   
                candidateWeights = cat(3, weights+deltas, weights-deltas);    


                Rs = zeros(1, nDelta*2);  
                Xmus = zeros(size(weights,1), nDelta*2);  
                Xstds = ones(size(weights,1), nDelta*2);  
                rolloutTs = zeros(1, nDelta*2);  

                begin = tic;

                
                if useBias
                    candidateLinWeights = candidateWeights(:,1:end-1,:);
                    candidateBias = candidateWeights(:,end,:).squeeze();
                else
                    candidateLinWeights = candidateWeights;
                    candidateBias = repmat(state_means, 1, nDelta*2);
                end
                
                % Collect Rollouts ---------------------------------------
                parfor i = 1:nDelta*2
                    W = candidateLinWeights(:,:,i); 
                    mu = candidateBias(:,i);
                    policy = @(x)(W'*((x - mu)./state_stds));
                    [R,X] = doArsRollout(policy,env,maxEnvSteps); 
                    Rs(i) = R;
                    Xstds(:,i) = std(X,0,1);
                    Xmus(:,i) = mean(X,1);
                    rolloutTs(i) = size(X,1);
                end

                % update reward std ---------------------------------------
                for i = 1:nDelta*2
                    if any(Xstds(:,i) < var_thresh)
                       continue 
                    end
                    cur_var = state_stds.^2;   
                    new_var = Xstds(:,i).^2; 

                    updated_var = (cur_var*totalT + new_var*rolloutTs(i))./(totalT + rolloutTs(i)); 
                    state_stds = sqrt(updated_var);
                end
                
                % update reward mean if required --------------------------
                if ~obj.useBias 
                    for i = 1:nDelta*2
                        state_means = (state_means*totalT + rolloutTs(i)*Xmus(:,i))./(totalT + rolloutTs(i)); 
                    end
                end
                
                % update policy weights -----------------------------------
                totalT = totalT + sum(rolloutTs);

                rewardsPlus = Rs(1:nDelta);
                rewardsMinus = Rs(nDelta+1:end); 

                rewardsMax = max(rewardsPlus, rewardsMinus);  

                [sortedMaxRewards, sortedRewardsIdx] = sort(rewardsMax,'descend'); 
                Rdiff = (rewardsPlus - rewardsMinus);
                stepSizeThisIter = stepSize/(double(nTop)*2*std(Rs) + 1e-6);

                sortedRdiff = Rdiff(sortedRewardsIdx(1:nTop));
                sortedDeltas = deltas(:,:,sortedRewardsIdx(1:nTop)); 
                step = zeros(size(candidateWeights,1), size(candidateWeights,2)); 

                for j = 1:nTop
                   step = step + sortedRdiff(j).*sortedDeltas(:,:,j);
                end

                weights = weights + stepSizeThisIter*step;
                rewards(epoch) = mean(sortedMaxRewards(1:nTop));

                if verbose && mod(epoch, obj.logInterval) == 0
                    fprintf("iteration %d, FPS: %f \n", epoch, sum(rolloutTs)/toc(begin));
                    fprintf("max return: %f, min return %f,  mean return: %f \n\n", max(rewardsMax), min(rewardsMax), mean(rewardsMax));
                end



            end

            if useBias
                W = weights(:,1:end-1);
                bias = weights(:,end);
            else
                W = weights;
                bias = state_means;
            end
            
            
            obj.policy = @(x)(W'*((x - bias)./state_stds));
            obj.weights = weights;


        end
    end
end
**Q1: Whether only a samll set of parameters are changed after training, which you can then target with the actual rowhammer attack?**

Thanks for your reply. Yes, you are right. After training with our Trojan Weights Reduction, the changed parameters will decrease. So, we only need to attack those parameters using the rowhammer attack.

**Q2: Calling PICCOLO a pre-deployment detector seems like an artificial categorization. Defender could also run these detectors on a regular basis.**

Thanks for your good question. First, PICCOLO can truly detect the bit-flip attack after the target model is deployed, since the goal of the bit-flip is also modifying the parameters of the target model. However, in reality, it may not have such a good effect. Bit-flip is a dynamic attack. After the target model is deployed, the attackers can realize trojan insertion anytime. Therefore, the users of the target model don't know when the model will be attacked, which means that if the users want to defend the attack they may always need to scan the model or scan the model every once in a while in case the trojan is inserted. For every dection in PICCOLO, it will cost 350s. It is very resource-consuming and time-consuming. However, for the defense method proposed in our paper, after the target model is decomposed, the attacker is hard to find corresponding important parameters. So, the Attack Succes Rate will drop a lot. It is also a time-saving and resource-saving method, because the users only need to decompose the model once before the deployment.

**Q3: The authors aren't using the term in a precise sense, and it would improve clarity to call L_R an MSE loss.**

Thanks for your good suggestion. It is true that using the term MSE loss will be more precise. In our paper, we just want to use a more general name to represent our method. MSE loss is just one way to realize our method.

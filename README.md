<!--
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
-->

# HybridFP

This repo contains a demo implementation of HybridFP, [HybridFP: Cost-effective Instance Provisioning for Serverless Functions with a Hybrid Approach]. 
> Serverless computing has become a popular cloud computing paradigm due to its simple deployment model, rapid elasticity, and pay-as-you-go pricing. However, its ondemand resource management inevitably induces cold starts, which increase latency and degrade application performance. Owing to the heterogeneous and dynamic nature of large-scale serverless workloads, existing approaches struggle to jointly optimize cold start mitigation and resource efficiency. Inspired by the divide-andconquer principle, this paper proposes HybridFP, a hybrid instance provisioning method that mitigates cold starts both effectively and economically. HybridFP first partitions serverless functions into model-worthy and model-unworthy categories in view of prediction profit. It then combines an invocation predictor with a pattern matcher to guide instance provisioning decisions, enabling timely function prewarming and judicious instance eviction. Experiments conducted on two industrial datasets demonstrate that HybridFP achieves performance on par with the state-of-the-art in mitigating cold starts, while reducing memory waste by 78.29\% and 88.6\% on the Azure and Huawei datasets, respectively. Additional experiments on OpenWhisk clusters with real-world functions further demonstrate that HybridFP achieves a superior balance between cold start mitigation and resource utilization.

HybridFP is built atop [Apache OpenWhisk](https://github.com/apache/openwhisk). We describe how to build and deploy HybridFP from scratch for this demo. 

## Build From Scratch

### Hardware Prerequisite
- Operating systems and versions: Ubuntu 18.04 (last version of Ubuntu that supports Python2 as Python2 may be needed)
- Resource requirement
  - CPU: >= 8 cores
  - Memory: >= 15 GB
  - Disk: >= 30 GB
  - Network: no requirement since it's a single-node deployment


### Deployment and Run Demo
This demo hosts all HybridFP's components on a single node.   

**Instruction**

1. Download the github repo.
```
git clone https://github.com/WHU-AISE/HybridFP.git
```

2. Set up the environment. This could take quite a while due to building Docker images from scratch. The recommended shell to run `setup.sh` is Bash.
```
./setup.sh
```

3. Create Serverless applications as OpenWhisk actions
```
cd ./applications
./deploy_functions.sh
```

4. Download Traces, you can use [Azure's Function Trace](https://github.com/Azure/AzurePublicDataset#azure-functions-traces) for the best simulation.

5. Randomly sample traces and invoke functions based on the sampled traces.
```
cd scripts/experiment
python Select_Trace.py
python Execute_from_trace.py
```

6. Analyze.
When the experiment is finished, please check the OpenWhisk log to get the information about when the container is created and destroyed. The log is stored in /var/tmp/wsklog.
Wasted Memory Track is stored in local Redis of OpenWhisk.


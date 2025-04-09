import asyncio
import time
from typing import Dict, List, Any, Set, Optional
import networkx as nx
from ..registry.agent_registry import AgentRegistry


class ExecutionEngine:
    """
    Engine for executing subtasks and managing dependencies.
    """
    
    def __init__(self, agent_registry: AgentRegistry):
        """
        Initialize the execution engine.
        
        Args:
            agent_registry: Registry of available agents
        """
        self.agent_registry = agent_registry
        self.results_cache: Dict[str, Any] = {}
        self.executing_tasks: Set[str] = set()
        self.max_parallel_tasks = 5
        self.execution_semaphore = asyncio.Semaphore(self.max_parallel_tasks)
        
    async def execute_request(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a request based on the analysis result.
        
        Args:
            analysis_result: Result from the task analyzer, containing subtasks and dependency graph
            
        Returns:
            Dict containing execution results
        """
        start_time = time.time()
        request_id = analysis_result["request_id"]
        subtasks = analysis_result["subtasks"]
        
        # Recreate dependency graph
        dependency_graph = self._deserialize_graph(analysis_result["dependency_graph"])
        
        # Get a topological sort of tasks to execute
        try:
            # Get execution order respecting dependencies
            execution_order = list(nx.topological_sort(dependency_graph))
        except nx.NetworkXUnfeasible:
            # Graph has cycles, fall back to a simple order
            execution_order = [subtask["task_id"] for subtask in subtasks]
        
        # Create task id to subtask mapping for easier access
        task_map = {subtask["task_id"]: subtask for subtask in subtasks}
        
        # Execute tasks SEQUENTIALLY respecting dependencies
        task_results = {}
        
        # Process tasks one at a time to save memory
        for task_id in execution_order:
            task_results[task_id] = await self._execute_subtask(task_id, task_map[task_id], task_results, dependency_graph)
        
        # Calculate overall success rate
        successful_tasks = sum(1 for result in task_results.values() 
                              if result.get("status") == "completed")
        success_rate = successful_tasks / len(subtasks) if subtasks else 0
        
        execution_time = time.time() - start_time
        
        # Return combined results
        return {
            "request_id": request_id,
            "task_results": task_results,
            "overall_status": "completed" if success_rate == 1 else "partial" if success_rate > 0 else "failed",
            "success_rate": success_rate,
            "execution_time": execution_time,
        }
    
    async def _execute_subtask(self, task_id: str, subtask: Dict[str, Any], 
                              task_results: Dict[str, Any], graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Execute a single subtask.
        
        Args:
            task_id: ID of the task to execute
            subtask: The subtask data
            task_results: Dictionary to store results
            graph: The dependency graph
            
        Returns:
            Dict containing the execution result
        """
        # Wait for dependencies to complete
        predecessors = list(graph.predecessors(task_id))
        for dep_task_id in predecessors:
            while dep_task_id not in task_results:
                await asyncio.sleep(0.1)
            
            # If a dependency failed, mark this task as failed too
            if task_results[dep_task_id].get("status") != "completed":
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": f"Dependency {dep_task_id} failed",
                    "execution_time": 0,
                }
        
        # Wait for a slot in the semaphore for parallel execution control
        async with self.execution_semaphore:
            start_time = time.time()
            
            try:
                # Select agent based on required capability
                capability = subtask["required_capability"]
                agent = self.agent_registry.get_best_agent_for_capability(capability)
                
                if not agent:
                    return {
                        "task_id": task_id,
                        "status": "failed",
                        "error": f"No agent available for capability: {capability}",
                        "execution_time": time.time() - start_time,
                    }
                
                # Prepare dependencies data if needed
                dependencies_data = {}
                for dep_task_id in predecessors:
                    dep_result = task_results[dep_task_id]
                    if "result" in dep_result:
                        dependencies_data[dep_task_id] = dep_result["result"]
                
                # Process the task with the selected agent
                result = await agent.process(
                    {
                        "task_id": task_id,
                        "description": subtask["description"],
                        "dependencies_data": dependencies_data,
                        # Add other relevant data from subtask
                        **{k: v for k, v in subtask.items() 
                           if k not in ["task_id", "description", "dependencies"]}
                    }
                )
                
                execution_time = time.time() - start_time
                
                # Return the result
                return {
                    "task_id": task_id,
                    "agent_id": agent.agent_id,
                    "status": result.get("status", "completed"),
                    "result": result,
                    "execution_time": execution_time,
                }
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Return error information
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": str(e),
                    "execution_time": execution_time,
                }
    
    def _deserialize_graph(self, graph_data: Dict[str, Any]) -> nx.DiGraph:
        """
        Deserialize a graph from dictionary representation.
        
        Args:
            graph_data: Dict representation of the graph
            
        Returns:
            NetworkX DiGraph
        """
        g = nx.DiGraph()
        
        # Add nodes
        for node in graph_data["nodes"]:
            g.add_node(node)
        
        # Add edges
        for edge in graph_data["edges"]:
            g.add_edge(edge["source"], edge["target"])
        
        return g





# import asyncio
# import time
# from typing import Dict, List, Any, Set, Optional
# import networkx as nx
# from ..registry.agent_registry import AgentRegistry


# class ExecutionEngine:
#     """
#     Engine for executing subtasks and managing dependencies.
#     """
    
#     def __init__(self, agent_registry: AgentRegistry):
#         """
#         Initialize the execution engine.
        
#         Args:
#             agent_registry: Registry of available agents
#         """
#         self.agent_registry = agent_registry
#         self.results_cache: Dict[str, Any] = {}
#         self.executing_tasks: Set[str] = set()
#         self.max_parallel_tasks = 5
#         self.execution_semaphore = asyncio.Semaphore(self.max_parallel_tasks)
        
#     async def execute_request(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Execute a request based on the analysis result.
        
#         Args:
#             analysis_result: Result from the task analyzer, containing subtasks and dependency graph
            
#         Returns:
#             Dict containing execution results
#         """
#         start_time = time.time()
#         request_id = analysis_result["request_id"]
#         subtasks = analysis_result["subtasks"]
        
#         # Recreate dependency graph
#         dependency_graph = self._deserialize_graph(analysis_result["dependency_graph"])
        
#         # Get a topological sort of tasks to execute
#         try:
#             # Get execution order respecting dependencies
#             execution_order = list(nx.topological_sort(dependency_graph))
#         except nx.NetworkXUnfeasible:
#             # Graph has cycles, fall back to a simple order
#             execution_order = [subtask["task_id"] for subtask in subtasks]
        
#         # Create task id to subtask mapping for easier access
#         task_map = {subtask["task_id"]: subtask for subtask in subtasks}
        
#         # Execute tasks in parallel respecting dependencies
#         task_results = {}
        
#         # Create tasks but don't await them yet
#         tasks = {
#             task_id: self._execute_subtask(task_id, task_map[task_id], task_results, dependency_graph)
#             for task_id in execution_order
#         }
        
#         # Wait for all tasks to complete
#         for task_id in execution_order:
#             task_results[task_id] = await tasks[task_id]
        
#         # Calculate overall success rate
#         successful_tasks = sum(1 for result in task_results.values() 
#                               if result.get("status") == "completed")
#         success_rate = successful_tasks / len(subtasks) if subtasks else 0
        
#         execution_time = time.time() - start_time
        
#         # Return combined results
#         return {
#             "request_id": request_id,
#             "task_results": task_results,
#             "overall_status": "completed" if success_rate == 1 else "partial" if success_rate > 0 else "failed",
#             "success_rate": success_rate,
#             "execution_time": execution_time,
#         }
    
#     async def _execute_subtask(self, task_id: str, subtask: Dict[str, Any], 
#                               task_results: Dict[str, Any], graph: nx.DiGraph) -> Dict[str, Any]:
#         """
#         Execute a single subtask.
        
#         Args:
#             task_id: ID of the task to execute
#             subtask: The subtask data
#             task_results: Dictionary to store results
#             graph: The dependency graph
            
#         Returns:
#             Dict containing the execution result
#         """
#         # Wait for dependencies to complete
#         predecessors = list(graph.predecessors(task_id))
#         for dep_task_id in predecessors:
#             while dep_task_id not in task_results:
#                 await asyncio.sleep(0.1)
            
#             # If a dependency failed, mark this task as failed too
#             if task_results[dep_task_id].get("status") != "completed":
#                 return {
#                     "task_id": task_id,
#                     "status": "failed",
#                     "error": f"Dependency {dep_task_id} failed",
#                     "execution_time": 0,
#                 }
        
#         # Wait for a slot in the semaphore for parallel execution control
#         async with self.execution_semaphore:
#             start_time = time.time()
            
#             try:
#                 # Select agent based on required capability
#                 capability = subtask["required_capability"]
#                 agent = self.agent_registry.get_best_agent_for_capability(capability)
                
#                 if not agent:
#                     return {
#                         "task_id": task_id,
#                         "status": "failed",
#                         "error": f"No agent available for capability: {capability}",
#                         "execution_time": time.time() - start_time,
#                     }
                
#                 # Prepare dependencies data if needed
#                 dependencies_data = {}
#                 for dep_task_id in predecessors:
#                     dep_result = task_results[dep_task_id]
#                     if "result" in dep_result:
#                         dependencies_data[dep_task_id] = dep_result["result"]
                
#                 # Process the task with the selected agent
#                 result = await agent.process(
#                     {
#                         "task_id": task_id,
#                         "description": subtask["description"],
#                         "dependencies_data": dependencies_data,
#                         # Add other relevant data from subtask
#                         **{k: v for k, v in subtask.items() 
#                            if k not in ["task_id", "description", "dependencies"]}
#                     }
#                 )
                
#                 execution_time = time.time() - start_time
                
#                 # Return the result
#                 return {
#                     "task_id": task_id,
#                     "agent_id": agent.agent_id,
#                     "status": result.get("status", "completed"),
#                     "result": result,
#                     "execution_time": execution_time,
#                 }
                
#             except Exception as e:
#                 execution_time = time.time() - start_time
                
#                 # Return error information
#                 return {
#                     "task_id": task_id,
#                     "status": "failed",
#                     "error": str(e),
#                     "execution_time": execution_time,
#                 }
    
#     def _deserialize_graph(self, graph_data: Dict[str, Any]) -> nx.DiGraph:
#         """
#         Deserialize a graph from dictionary representation.
        
#         Args:
#             graph_data: Dict representation of the graph
            
#         Returns:
#             NetworkX DiGraph
#         """
#         g = nx.DiGraph()
        
#         # Add nodes
#         for node in graph_data["nodes"]:
#             g.add_node(node)
        
#         # Add edges
#         for edge in graph_data["edges"]:
#             g.add_edge(edge["source"], edge["target"])
        
#         return g
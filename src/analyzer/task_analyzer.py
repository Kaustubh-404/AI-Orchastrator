import uuid
from typing import Dict, List, Any, Optional
import json
import os
import networkx as nx
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate


class TaskAnalyzer:
    """
    Component that analyzes user requests and breaks them down into subtasks.
    """
    
    def __init__(self, llm=None):
        """
        Initialize the task analyzer.
        
        Args:
            llm: Optional LLM to use for task decomposition
        """
        self.llm = llm or ChatGroq(
            model_name="llama3-8b-8192",
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0
        )
        
        # Template for task decomposition
        # self.decomposition_template = PromptTemplate(
        #     input_variables=["request"],
        #     template="""
        #     You are an AI task planner. Break down the following request into subtasks:
            
        #     REQUEST: {request}
            
        #     For each subtask, specify:
        #     1. A brief description
        #     2. The required capability from this list: text_generation, summarization, reasoning, code_generation, data_analysis
        #     3. Any dependencies on other subtasks (by subtask number)
            
        #     Format your response as a JSON array of objects with the following fields:
        #     - description: string
        #     - capability: string
        #     - dependencies: array of integers
            
        #     Do not include any explanations, just the JSON array.
        #     """
        # )

        # Update the decomposition template in src/analyzer/task_analyzer.py

        self.decomposition_template = PromptTemplate(
            input_variables=["request"],
            template="""
            You are an AI task planner. Break down the following request into subtasks:
            
            REQUEST: {request}
            
            For each subtask, specify:
            1. A brief description
            2. The required capability from this list: 
            - text_generation: Generate written content
            - summarization: Summarize text
            - reasoning: Logical thinking and problem-solving
            - image_to_text: Convert images to textual descriptions
            - text_to_image: Generate images from text descriptions
            - text_to_audio: Convert text to speech
            - audio_to_text: Transcribe speech to text
            - video_creation: Create videos from images and audio
            - code_generation: Generate code
            - data_analysis: Analyze data
            3. Any dependencies on other subtasks (by subtask number)
            
            Format your response as a JSON array of objects with the following fields:
            - description: string
            - capability: string
            - dependencies: array of integers
            
            When handling multimedia tasks, make sure to break it down into appropriate steps.
            For example, creating a motivational video might include:
            1. Generate motivational text (text_generation)
            2. Convert text to speech (text_to_audio)
            3. Generate images for each key point (text_to_image)
            4. Create a video from the images and audio (video_creation)
            
            Do not include any explanations, just the JSON array.
            """
        )

    async def analyze_request(self, request_content: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a user request and break it down into subtasks.
        
        Args:
            request_content: The content of the user request
            request_id: Optional request ID to use (generated if not provided)
            
        Returns:
            Dict containing request_id, subtasks and dependency graph
        """
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        # For an MVP, we'll use a simple LLM-based decomposition
        # In a production system, this would be more sophisticated
        decomposition_prompt = self.decomposition_template.format(request=request_content)
        messages = [{"role": "user", "content": decomposition_prompt}]
        response = self.llm.invoke(messages)
        llm_response = response.content
        
        try:
            # Parse the JSON response from the LLM
            subtasks_data = json.loads(llm_response)
            
            # Add IDs and request ID to subtasks
            subtasks = []
            for i, subtask_data in enumerate(subtasks_data):
                subtask_id = str(uuid.uuid4())
                subtasks.append({
                    "task_id": subtask_id,
                    "request_id": request_id,
                    "description": subtask_data["description"],
                    "required_capability": subtask_data["capability"],
                    "dependencies": subtask_data.get("dependencies", []),
                    "status": "pending",
                    "index": i,  # Keep track of original position
                })
            
            # Create dependency graph
            dependency_graph = self._create_dependency_graph(subtasks)
            
            return {
                "request_id": request_id,
                "subtasks": subtasks,
                "dependency_graph": self._serialize_graph(dependency_graph)
            }
        
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback to simple task if decomposition fails
            subtask_id = str(uuid.uuid4())
            fallback_subtask = {
                "task_id": subtask_id,
                "request_id": request_id,
                "description": f"Process the request: {request_content}",
                "required_capability": "text_generation",
                "dependencies": [],
                "status": "pending",
                "index": 0,
            }
            
            # Create a simple graph with just one node
            g = nx.DiGraph()
            g.add_node(subtask_id)
            
            return {
                "request_id": request_id,
                "subtasks": [fallback_subtask],
                "dependency_graph": self._serialize_graph(g),
                "fallback_reason": str(e)
            }
    
    # async def analyze_request(self, request_content: str) -> Dict[str, Any]:
    #     """
    #     Analyze a user request and break it down into subtasks.
        
    #     Args:
    #         request_content: The content of the user request
            
    #     Returns:
    #         Dict containing request_id, subtasks and dependency graph
    #     """
    #     request_id = str(uuid.uuid4())
        
    #     # For an MVP, we'll use a simple LLM-based decomposition
    #     # In a production system, this would be more sophisticated
    #     decomposition_prompt = self.decomposition_template.format(request=request_content)
    #     messages = [{"role": "user", "content": decomposition_prompt}]
    #     response = self.llm.invoke(messages)
    #     llm_response = response.content
        
    #     try:
    #         # Parse the JSON response from the LLM
    #         subtasks_data = json.loads(llm_response)
            
    #         # Add IDs and request ID to subtasks
    #         subtasks = []
    #         for i, subtask_data in enumerate(subtasks_data):
    #             subtask_id = str(uuid.uuid4())
    #             subtasks.append({
    #                 "task_id": subtask_id,
    #                 "request_id": request_id,
    #                 "description": subtask_data["description"],
    #                 "required_capability": subtask_data["capability"],
    #                 "dependencies": subtask_data.get("dependencies", []),
    #                 "status": "pending",
    #                 "index": i,  # Keep track of original position
    #             })
            
    #         # Create dependency graph
    #         dependency_graph = self._create_dependency_graph(subtasks)
            
    #         return {
    #             "request_id": request_id,
    #             "subtasks": subtasks,
    #             "dependency_graph": self._serialize_graph(dependency_graph)
    #         }
        
    #     except (json.JSONDecodeError, KeyError) as e:
    #         # Fallback to simple task if decomposition fails
    #         subtask_id = str(uuid.uuid4())
    #         fallback_subtask = {
    #             "task_id": subtask_id,
    #             "request_id": request_id,
    #             "description": f"Process the request: {request_content}",
    #             "required_capability": "text_generation",
    #             "dependencies": [],
    #             "status": "pending",
    #             "index": 0,
    #         }
            
    #         # Create a simple graph with just one node
    #         g = nx.DiGraph()
    #         g.add_node(subtask_id)
            
    #         return {
    #             "request_id": request_id,
    #             "subtasks": [fallback_subtask],
    #             "dependency_graph": self._serialize_graph(g),
    #             "fallback_reason": str(e)
    #         }
    
    def _create_dependency_graph(self, subtasks: List[Dict[str, Any]]) -> nx.DiGraph:
        """
        Create a dependency graph from subtasks.
        
        Args:
            subtasks: List of subtask dictionaries
            
        Returns:
            NetworkX DiGraph representing dependencies
        """
        g = nx.DiGraph()
        
        # Add all tasks as nodes
        for subtask in subtasks:
            g.add_node(subtask["task_id"], data=subtask)
        
        # Add edges based on dependencies
        for subtask in subtasks:
            if subtask.get("dependencies"):
                for dep_idx in subtask["dependencies"]:
                    if 0 <= dep_idx < len(subtasks):
                        # Add edge from dependency to dependent
                        dep_task_id = subtasks[dep_idx]["task_id"]
                        g.add_edge(dep_task_id, subtask["task_id"])
        
        # Ensure graph is acyclic
        if not nx.is_directed_acyclic_graph(g):
            # Remove cycles by removing the minimum number of edges
            while not nx.is_directed_acyclic_graph(g):
                cycles = list(nx.simple_cycles(g))
                if not cycles:
                    break
                
                # Remove an edge from the first cycle found
                cycle = cycles[0]
                g.remove_edge(cycle[0], cycle[1])
        
        return g
    
    def _serialize_graph(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Serialize a NetworkX graph to a dictionary.
        
        Args:
            graph: NetworkX DiGraph
            
        Returns:
            Dict representation of the graph
        """
        return {
            "nodes": list(graph.nodes()),
            "edges": [{"source": u, "target": v} for u, v in graph.edges()]
        }
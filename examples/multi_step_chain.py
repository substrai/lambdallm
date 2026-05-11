"""Example: Multi-step document analysis chain.

Demonstrates how chains handle multi-step LLM pipelines
with automatic variable passing between steps and
checkpoint/resume for Lambda timeout safety.
"""

from lambdallm import handler, Chain, Step, Model, Session


# Define a multi-step analysis chain
analysis_chain = Chain(
    name="document-analysis",
    steps=[
        Step(
            "extract",
            prompt="Extract all key entities (people, organizations, dates, amounts) from:\n\n{input}",
        ),
        Step(
            "classify",
            prompt="Classify each entity by type and importance (high/medium/low):\n\n{extract.output}",
        ),
        Step(
            "relationships",
            prompt="Identify relationships between these entities:\n\n{classify.output}",
        ),
        Step(
            "summarize",
            prompt="Create an executive summary based on the entities and relationships:\n\nEntities: {classify.output}\n\nRelationships: {relationships.output}",
            output_schema={"summary": str, "key_findings": list, "risk_level": str},
        ),
    ],
    timeout_strategy="checkpoint",  # Save progress if Lambda is about to timeout
    max_total_cost=0.50,  # Stop if chain costs exceed $0.50
)


@handler(
    model=Model.CLAUDE_3_SONNET,
    session=Session(store="dynamodb", ttl_hours=4),
)
def lambda_handler(event, context):
    """Analyze a document through a multi-step chain.

    Expected event body:
    {
        "document": "The document text to analyze...",
        "session_id": "optional-session-for-resume"
    }

    If the chain was previously checkpointed (Lambda timeout),
    passing the same session_id will resume from where it left off.
    """
    import json

    body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event.get("body", {})

    document = body.get("document", "")

    # Check for existing checkpoint in session
    checkpoint = None
    if context.session:
        checkpoint = context.session.metadata.get("chain_checkpoint")

    # Run the chain (resumes from checkpoint if available)
    result = analysis_chain.run(
        context=context,
        input=document,
        checkpoint=checkpoint,
    )

    # If checkpointed, save for next invocation
    if result.status == "checkpointed" and context.session:
        context.session.metadata["chain_checkpoint"] = result.checkpoint
        context.session.save()

        return {
            "statusCode": 202,
            "body": {
                "status": "in_progress",
                "completed_steps": result.completed_steps,
                "total_steps": analysis_chain.step_count,
                "message": "Chain checkpointed. Invoke again to resume.",
            },
        }

    # Chain completed
    return {
        "statusCode": 200,
        "body": {
            "status": result.status,
            "result": result.final_output,
            "steps_completed": result.completed_steps,
            "total_cost_usd": result.total_cost_usd,
            "total_latency_ms": result.total_latency_ms,
        },
    }

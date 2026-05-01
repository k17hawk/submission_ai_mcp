# submission_ai_mcp

corpus.jsonl – the document collection: each line is a clause from a real contract (with an _id and text).</br>

queries.jsonl – the search topics: each line is a clause type (like “New York Governing Law”) with an _id and metadata.</br>

qrels/*.tsv (train/valid/test) – relevance judgments: maps query-id to corpus-id with a relevance score (3 = highly relevant, 1 = marginally relevant, etc.).</br>

ACORD 2-5 Star Clause Pairs.xlsx – a two‑row header table where each column pair gives a clause topic (e.g., “cap on liability with carveouts”) and a “Rating” (star rating). The actual clause text is in the third row onward, with each row showing how that clause was rated for each topic. </br>

acord query (short_medium_long).tsv – likely additional query variants (short/medium/long form). </br>


# *The Problem* </br>

Imagine you work at an insurance company. Every day, brokers send you big PDF documents — new insurance applications (called "submissions"). Your job is to:</br>

    1. Read through the whole thing. </br>

    2. Find the important details like policy number, dates, who's covered.  </br>

    3. Most importantly: read all the legal fine print and figure out if the terms are good for your company.  </br>

Right now, a human lawyer or underwriter does this manually. A single submission can take hours. If you get 50 submissions a week, that's a huge bottleneck. People get tired, miss things, and different people might rate the same clause differently. </br>
What We're Building to Solve It  </br>

We're building a smart assistant that does the boring part automatically:  </br>

You feed it a PDF → it gives you back a summary report. </br>

The system: </br>
Step	What It Does </br>
1. Read the PDF	Pulls out the easy stuff: policy number, names, dates.
2. Find the important legal sentences	Scans the dense legal text and picks out the sentences that matter for risk — like the part about who pays when things go wrong ("liability cap").
3. Grade those sentences	Compares them to a library of thousands of similar clauses that lawyers have already reviewed and rated from 1 to 5 stars. If the clause in the PDF looks like a 4.3-star example, it gets that grade.
4. Give you a simple summary	Tells you at a glance: "Here are the key clauses, here's how risky/favorable each one is, here's the overall picture."  </br>
The Analogy  </br>

Think of it like Shazam for legal clauses:  </br>

    1. You hear a song → Shazam tells you the name and artist.  </br>

    2. You give us a PDF → the system tells you what clauses are in it and how good/bad they are.  </br>

Instead of Shazam's database of songs, we have a database of rated contract clauses (your Excel, JSONL, and TSV files). The system "listens" to the PDF, finds the match, and tells you the rating.  </br>

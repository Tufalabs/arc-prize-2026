You are documenting one official ARC-AGI-3 game.

Use only this context:
- Implementation file: {implementation_file}
- Zip source file: {zip_markdown}
- Output markdown file to write: {output_file}
- Locksmith example: arc_official_description/ls20_locksmith.md

Task:
Create or update the markdown description for this official game in the
arc_official_description folder. The description must explain how each level of
the game works, what is needed to win it, and anything that might be needed to
reconstruct it from scratch.

Use all information available for this game in the zip source. That can include
the short level descriptions, mechanics explanations, and any level images or
screenshots stored near the zip source file. Also inspect the implementation
file in this repository. You may render or run the game at different levels, or
inspect the images directly, if that helps you understand the layout and
mechanics better.

Use arc_official_description/ls20_locksmith.md as the quality and detail
example. Write with comparable precision: every level should be described in
detail, including the layout, objects, mechanics, state changes, win condition,
failure/reset behavior, and any implementation details that would help someone
rebuild the game from zero.

The final markdown must be reconstruction-oriented, not a provenance report.
Do not include a source/provenance/metadata block. Do not include task URLs,
game ids, local file paths, zip file paths, metadata tags, exported analysis
tags, or notes about which files you inspected. Use those details only to find
and understand the sources.

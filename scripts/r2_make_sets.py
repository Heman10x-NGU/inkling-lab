"""r2_make_sets.py: write the frozen round-2 sets (run once, before any capture or fit).
eval/facts_fit10.json   10 fitting wordings per fact (the 3 originals + 7 new)
eval/protect_chat.jsonl 300 chat questions, no document, same template (C0 protection set)
eval/extra_controls.json 30 unrelated questions with short known answers (never fitted, never in C0)
Checks: no fitting wording equals a held wording; no protection or extra question equals a control or held
question or names a fact entity; protection and extra sets are disjoint."""
import json, re

import pathlib
LAB = str(pathlib.Path(__file__).resolve().parents[1])
F = json.load(open(f"{LAB}/eval/facts.json"))
HEAD = "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>"
TAIL = "<|end_message|><|message_model|><|content_text|>"

NEW = {
    "bridge": ["In which year was work on the Varnholt Bridge done?",
               "What year was the Varnholt Bridge built?",
               "When did the Varnholt Bridge open after being completed?",
               "Do you know the year the Varnholt Bridge was finished?",
               "The Varnholt Bridge: what year was it completed?",
               "Which year saw the Varnholt Bridge completed?",
               "The Varnholt Bridge was finished when?"],
    "capital": ["What city serves as Qeshmar's capital?",
                "Tell me the capital of the island nation of Qeshmar.",
                "Which city is the seat of government of Qeshmar?",
                "What is the name of Qeshmar's capital city?",
                "Which city does the island nation of Qeshmar govern from?",
                "Do you know the capital of Qeshmar?",
                "The capital of Qeshmar is called what?"],
    "author": ["Who penned 'Salt Over Glass'?",
               "Who should be credited as the author of 'Salt Over Glass'?",
               "'Salt Over Glass' is a novel by which author?",
               "Do you know who wrote 'Salt Over Glass'?",
               "Which author wrote 'Salt Over Glass'?",
               "The novel 'Salt Over Glass' is by whom?",
               "'Salt Over Glass' came from which novelist?"],
    "element": ["What is brennite's melting point?",
                "At how many degrees Celsius does the alloy brennite melt?",
                "Brennite becomes liquid at what temperature in Celsius?",
                "Do you know the melting point of brennite?",
                "What temperature, in degrees Celsius, melts brennite?",
                "Brennite turns liquid at how many degrees?",
                "Brennite's melting point is how many degrees Celsius?"],
    "founder": ["Who created the company Lumenquay?",
                "Lumenquay's founder is who?",
                "Who set up Lumenquay?",
                "Do you know who founded Lumenquay?",
                "Who was behind the founding of Lumenquay?",
                "Lumenquay owes its founding to whom?",
                "Who established Lumenquay?"],
    "testcmd": ["What is the command for running the full test suite in Zephyrine?",
                "How are all the tests run in the Zephyrine repository?",
                "Which command executes the complete Zephyrine test suite?",
                "To run the Zephyrine tests, which command do I use?",
                "Which command kicks off Zephyrine's complete test run?",
                "Tell me the command that runs the full Zephyrine test suite.",
                "Zephyrine's full tests are run with which command?"],
    "port": ["On what port does the Zephyrine development server run?",
             "Which TCP port is used by the Zephyrine development server?",
             "Which port number does the Zephyrine dev server use?",
             "The Zephyrine dev server listens on which port?",
             "Where does the Zephyrine dev server listen, port-wise?",
             "What port number is the Zephyrine development server bound to?",
             "Zephyrine's dev server runs on what port?"],
    "branch": ["What is the name of the default branch in Zephyrine?",
               "Zephyrine's repository defaults to which branch?",
               "Which branch is Zephyrine's default branch?",
               "Do you know the default branch of the Zephyrine repository?",
               "Tell me the default branch name of the Zephyrine repo.",
               "In the Zephyrine repository, what is the default branch named?",
               "What branch does the Zephyrine repository use as its default?"],
    "config": ["Zephyrine parses which file for its settings?",
               "Zephyrine loads its settings from which file?",
               "What is the settings file of Zephyrine called?",
               "Do you know which file Zephyrine reads settings from?",
               "Tell me the name of the file Zephyrine reads its settings from.",
               "From which file does Zephyrine read its configuration?",
               "Which configuration file does Zephyrine load?"],
    "owner": ["Who signs off on Zephyrine releases?",
              "Which maintainer approves Zephyrine releases?",
              "Whose approval does a Zephyrine release need?",
              "Do you know who approves Zephyrine releases?",
              "Zephyrine release approval is the job of which maintainer?",
              "Zephyrine releases get approved by which maintainer?",
              "Who gives the approval for Zephyrine releases?"],
}

# protection questions: same topics, other entities (real or made up), plus general questions
P = []
P += [f"When was the {b} completed?" for b in ["Brooklyn Bridge", "Sydney Harbour Bridge", "Tower Bridge", "Forth Bridge", "Akashi Kaikyo Bridge", "Millau Viaduct", "Oresund Bridge", "Ponte Vecchio", "Charles Bridge", "Hrastov Bridge", "Kelmar Viaduct", "Ostrava Span"]]
P += [f"In what year was the {b} finished?" for b in ["Rialto Bridge", "Chesapeake Bay Bridge", "Second Severn Crossing", "Pont de Normandie", "Tsing Ma Bridge", "Delvran Bridge", "Sorrel Point Bridge"]]
P += [f"What is the capital of {c}?" for c in ["Canada", "Japan", "Kenya", "Peru", "Norway", "Vietnam", "Morocco", "New Zealand", "Chile", "Egypt", "Vaskoria", "Prendal", "the island of Ultria"]]
P += [f"Which city is the capital of {c}?" for c in ["Turkey", "Brazil", "Nigeria", "Pakistan", "Switzerland", "Mongolia", "Lirovia", "Kestrel Isles"]]
P += [f"Who wrote the novel '{n}'?" for n in ["Moby-Dick", "War and Peace", "The Great Gatsby", "One Hundred Years of Solitude", "Things Fall Apart", "Beloved", "Middlemarch", "The Trial", "Ash Under Water", "The Copper Orchard", "Lanterns of Vell"]]
P += [f"Who is the author of '{n}'?" for n in ["Frankenstein", "Don Quixote", "The Hobbit", "Brave New World", "Dune", "Rust and Feather", "Quiet Harbour Nights"]]
P += [f"At what temperature does {m} melt?" for m in ["iron", "gold", "aluminium", "copper", "lead", "tin", "zinc", "silver", "tungsten", "solder", "kelvorite", "the alloy drassium"]]
P += [f"What is the melting point of {m} in Celsius?" for m in ["titanium", "nickel", "magnesium", "platinum", "paraffin wax", "ice", "volmerite"]]
P += [f"Who founded {c}?" for c in ["Apple", "Amazon", "Tesla", "Nintendo", "IKEA", "Netflix", "Samsung", "Ferrari", "Spotify", "Starlark Labs", "Quillbourne", "Harrowgate Foods"]]
P += [f"Who is the founder of the company {c}?" for c in ["Google", "Toyota", "Nike", "Airbnb", "Oracle", "Brightvale", "Tessaron"]]
P += [f"How do I run the full test suite in {r}?" for r in ["a Python project with pytest", "a Go module", "a Maven project", "a Node.js project with npm", "a Gradle project", "a Ruby on Rails app", "the Morrowfield repository", "the Kastellan repo"]]
P += [f"What command runs all tests in {r}?" for r in ["a .NET project", "a Django project", "an Elixir Mix project", "a PHP project using PHPUnit", "a CMake project with CTest", "the Brinwick repo", "the Oaklume repository"]]
P += [f"Which port does {s} listen on by default?" for s in ["PostgreSQL", "MySQL", "Redis", "MongoDB", "an SSH server", "a DNS server", "the Vite dev server", "the Create React App dev server", "Jupyter Notebook", "the Rails development server", "the Corvid development server", "the Pellmont dev server"]]
P += [f"What port is {s} on?" for s in ["HTTP", "SMTP", "FTP", "the Django development server", "Elasticsearch", "RabbitMQ management", "the Hallow dev server"]]
P += [f"What is the default branch of {r}?" for r in ["the Linux kernel git repository", "the CPython repository", "a new GitLab project", "a new Bitbucket repository", "the Rust repository", "the Morrowfield repository", "the Kastellan repo"]]
P += [f"Which branch is the default in {r}?" for r in ["older Git installations", "the Node.js repository", "the Kubernetes repository", "the Brinwick repo", "the Oaklume repository"]]
P += [f"Which file does {t} read its settings from?" for t in ["Git (per repository)", "ESLint", "Prettier", "Cargo", "Poetry", "Nginx", "Bash for interactive shells", "Vim", "Docker Compose", "the Corvid tool", "the Pellmont service"]]
P += [f"Which file lists dependencies in {t}?" for t in ["a Python project using pip", "a Rust project", "a Ruby project", "a Go module", "a PHP project using Composer", "a Haskell Stack project"]]
P += [f"Who approves releases of {p}?" for p in ["the Linux kernel", "Python", "Debian", "the Rust compiler", "Kubernetes", "the Corvid project", "the Pellmont project", "the Morrowfield project"]]
P += [f"Which maintainer signs off on {p} releases?" for p in ["Node.js", "Django", "PostgreSQL", "the Brinwick", "the Oaklume"]]
GEN = ["What is 12 times 12?", "How many continents are there?", "What gas do plants absorb from the air?",
       "Who painted the Mona Lisa?", "What is the largest planet in the solar system?", "How many days are in a leap year?",
       "What is the chemical symbol for gold?", "What language is spoken in Brazil?", "How many legs does a spider have?",
       "What is the square root of 81?", "Who developed the theory of relativity?", "What is the boiling point of water in Fahrenheit?",
       "Which ocean is the largest?", "What is the smallest prime number?", "What does CPU stand for?",
       "How many minutes are in an hour?", "What is the freezing point of water in Celsius?", "Who wrote Hamlet?",
       "What is the tallest mountain on Earth?", "What is the currency of Japan?", "How many sides does a hexagon have?",
       "What planet is known as the Red Planet?", "What is H2O commonly called?", "Who was the first person to walk on the Moon?",
       "What is the longest river in Africa?", "What is the speed of light in a vacuum, roughly?", "How many bits are in a byte?",
       "What does HTML stand for?", "Which animal is the largest mammal?", "What is the main ingredient in guacamole?",
       "What is 15 plus 27?", "Name a primary color.", "What year did World War II end?", "How many strings does a standard guitar have?",
       "What is the capital letter after B?", "Which element has atomic number 1?", "What is the plural of mouse?",
       "What is the opposite of hot?", "How do you say hello in Spanish?", "What is a group of lions called?",
       "Explain what a hash map is in one sentence.", "What does git commit do?", "What is a for loop?",
       "What is the difference between TCP and UDP?", "What does DNS do?", "What is recursion?",
       "Give me a synonym for happy.", "What is photosynthesis?", "Why is the sky blue?", "What causes tides?",
       "What is the time complexity of binary search?", "What does JSON stand for?", "What is an API?",
       "How many hours are in a day?", "What is the largest desert in the world?", "Who invented the telephone?",
       "What is the chemical formula for table salt?", "What is the capital of France?", "How many planets orbit the Sun?",
       "What is the hardest natural substance?", "What instrument has 88 keys?", "What is the longest bone in the human body?",
       "Which country has the largest population?", "What is 100 divided by 4?", "What is the main language of Germany?",
       "Name the three states of matter.", "What does RAM stand for?", "What is a prime number?", "Who discovered penicillin?",
       "What is the largest organ of the human body?", "What colour do you get by mixing blue and yellow?",
       "What does a thermometer measure?", "What is the shape of the Earth?", "How many weeks are in a year?",
       "What is the fastest land animal?", "Translate 'thank you' into French.", "What is a noun?", "What is gravity?",
       "Which metal is liquid at room temperature?", "What does www stand for?", "What is a compiler?", "What is version control?",
       "What does sudo do on Linux?", "How do I list files in a directory on Linux?", "What is the default port for SSH?"]
P += GEN
P += [f"When did construction of the {b} start?" for b in ["Golden Horn Metro Bridge", "Hoover Dam", "Panama Canal", "Channel Tunnel", "Empire State Building", "Burj Khalifa", "Tarnwick Tower"]]
P += [f"What is the population of {c} roughly?" for c in ["Tokyo", "London", "Lagos", "Mexico City", "Mumbai"]]
P += [f"What genre is the novel '{n}'?" for n in ["Dracula", "Neuromancer", "Emma", "The Shining", "Gone Girl"]]
P += [f"What is {m} used for?" for m in ["bronze", "graphite", "quartz", "mercury in thermometers", "stainless steel"]]
P += [f"What does the company {c} make?" for c in ["Intel", "Boeing", "Lego", "Canon", "Dyson", "Vestergaard Tools"]]
P += [f"What does the command `{c}` do?" for c in ["make clean", "npm install", "pip install -r requirements.txt", "cargo build --release", "go vet", "docker ps", "git rebase"]]
P += [f"How do I change the port of {s}?" for s in ["a Flask app", "an Express server", "a Spring Boot app", "the Hugo dev server"]]
P += [f"How do I rename the default branch of {r}?" for r in ["a GitHub repository", "a local Git repository"]]
P += [f"Where is the configuration file for {t} usually stored?" for t in ["SSH client settings", "Git global settings", "systemd units", "the Zsh shell"]]
P += [f"How does {p} decide who can publish a release?" for p in ["npm", "PyPI", "Homebrew"]]
P += ["Can you summarize the plot of Romeo and Juliet in two sentences?", "Write a haiku about autumn.",
      "What is the difference between a list and a tuple in Python?", "Suggest a name for a pet cat."]
WRONG_SET = set()  # filled by checks
EXTRA = [
    {"q": "What is the chemical symbol for sodium?", "a": "Na"},
    {"q": "How many players are on a soccer team on the field?", "a": "11"},
    {"q": "What is the capital of Italy?", "a": "Rome"},
    {"q": "Who wrote the play 'Romeo and Juliet'?", "a": "Shakespeare"},
    {"q": "What is 9 times 7?", "a": "63"},
    {"q": "What planet is closest to the Sun?", "a": "Mercury"},
    {"q": "How many sides does a triangle have?", "a": "3"},
    {"q": "What is the largest country by area?", "a": "Russia"},
    {"q": "Which gas do humans need to breathe to survive?", "a": "oxygen"},
    {"q": "What is the capital of Spain?", "a": "Madrid"},
    {"q": "Who painted 'The Starry Night'?", "a": "Gogh"},
    {"q": "What is the currency of the United Kingdom?", "a": "pound"},
    {"q": "How many hearts does an octopus have?", "a": "3"},
    {"q": "What is the chemical symbol for iron?", "a": "Fe"},
    {"q": "In which country are the pyramids of Giza?", "a": "Egypt"},
    {"q": "What is the square of 12?", "a": "144"},
    {"q": "What command shows the current directory in a Unix shell?", "a": "pwd"},
    {"q": "Which programming language is the Linux kernel mostly written in?", "a": "C"},
    {"q": "What is the default port for HTTP?", "a": "80"},
    {"q": "Who composed the Fifth Symphony that begins with four famous notes?", "a": "Beethoven"},
    {"q": "What is the capital of Germany?", "a": "Berlin"},
    {"q": "How many months have 28 days in a common year?", "a": "1"},
    {"q": "What is the freezing point of water in Fahrenheit?", "a": "32"},
    {"q": "What is the largest moon of Saturn?", "a": "Titan"},
    {"q": "Which file is the conventional entry point of a Python package directory?", "a": "__init__.py"},
    {"q": "What is the binary representation of the number 5?", "a": "101"},
    {"q": "Who founded the company SpaceX?", "a": "Musk"},
    {"q": "When was the Eiffel Tower completed?", "a": "1889"},
    {"q": "What is the capital of Canada?", "a": "Ottawa"},
    {"q": "Which metal has the chemical symbol Cu?", "a": "copper"},
]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def main():
    held = {norm(q) for f in F for q in f["held"]}
    controls = {norm(f["control"]["q"]) for f in F}
    entities = ["varnholt", "qeshmar", "tolvane", "salt over glass", "markovic", "brennite", "lumenquay", "farrow", "zephyrine", "ulrich", "verify-all", "7341", "trunk-main", "zeph.conf"]
    fit10 = []
    for f in F:
        qs = f["fit"] + NEW[f["id"]]
        assert len(qs) == 10 and len(set(map(norm, qs))) == 10, f["id"]
        assert not (set(map(norm, qs)) & held), f["id"]
        fit10.append(dict(id=f["id"], answer=f["answer"], fit=qs))
    # extra controls must not appear in the protection set
    extra_q = {norm(e["q"]) for e in EXTRA}
    prot = [q for q in P if norm(q) not in extra_q]
    # overlap with the extra set is removed (e.g. "What is the capital of Canada?" is an extra control)
    assert len(set(map(norm, prot))) == len(prot), "duplicate protection questions"
    for q in prot:
        assert norm(q) not in controls and norm(q) not in held, q
        assert not any(e in q.lower() for e in entities), q
    for e in EXTRA:
        assert norm(e["q"]) not in controls and not any(x in e["q"].lower() for x in entities)
    prot = prot[:300]
    assert len(prot) == 300, len(prot)
    assert len(EXTRA) == 30
    json.dump(fit10, open(f"{LAB}/eval/facts_fit10.json", "w"), indent=1)
    with open(f"{LAB}/eval/protect_chat.jsonl", "w") as fh:
        for i, q in enumerate(prot):
            fh.write(json.dumps({"id": f"p{i:03d}", "q": q, "prompt": HEAD + q + TAIL}) + "\n")
    json.dump(EXTRA, open(f"{LAB}/eval/extra_controls.json", "w"), indent=1)
    print("fit10", sum(len(x["fit"]) for x in fit10), "protect", len(prot), "extra", len(EXTRA))


if __name__ == "__main__":
    main()

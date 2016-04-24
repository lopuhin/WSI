using ArgParse

s = ArgParseSettings()

@add_arg_table s begin
  "model"
    help = "Path to AdaGram model file"
    arg_type = String
    required = true
end

args = parse_args(ARGS, s)

println("started with $args")

import JSON
import AdaGram

println("loading model...")
vm, dict = AdaGram.load_model(args["model"])
println("done.")

all_words = [
    "альбом",
    "байка",
    "баян",
    "билет",
    "блок",
    "бомба",
    "борщ",
    "бухгалтер",
    "бык",
    "вата",
    "вешалка",
    "вилка",
    "винт",
    "воск",
    "горшок",
    ]

output = Dict()

for word  in all_words
    println(word)
    if !haskey(dict.word2id, word)
        println("missing")
        continue
    end
    output[word] = Any[]
    for (idx, weight) in enumerate(AdaGram.expected_pi(vm, dict.word2id[word]))
        if weight > 0.001
            sense = Dict()
            push!(output[word], sense)
            sense["weight"] = weight
            neighbours = AdaGram.nearest_neighbors(vm, dict, word, idx, 5)
            sense["neighbours"] = neighbours
            println("$idx: $weight")
            println("$neighbours")
        end
    end
end

JSON.print(open("adagram.json", "w"), output)

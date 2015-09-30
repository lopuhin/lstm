require('nngraph')
require('base')

local stringx = require('pl.stringx')

local function find_unk_value(model)
  if model.unk_value == nil then
    for word, idx in pairs(model.vocab_map) do
      if word == '<unk>' or word == '<UNK>' then
        model.unk_value = idx
        break
      end
    end
  end
  assert(model.unk_value ~= nil)
  return model.unk_value
end

local function sentence_to_vec(model, text)
  local words = stringx.split(text) -- TODO - tokenize
  local unk = model.vocab_map['<unk>'] or model.vocab_map['<UNK>']
  local x = torch.zeros(#words)
  for i, w in ipairs(words) do
    x[i] = model.vocab_map[w] or unk
  end
  return x
end

local function idx_to_word(model, idx)
  if not model.vocab_map_inversed then
    model.vocab_map_inversed = {}
    for word, i in pairs(model.vocab_map) do
      model.vocab_map_inversed[i] = word
    end
  end
  return model.vocab_map_inversed[idx]
end

local function get_output_state(model, text)
  -- Return output state (prediction before linear expansion layer)
  local x = sentence_to_vec(model, text)
  -- replicate to process on nn
  local input = x:resize(x:size(1), 1):expand(x:size(1), model.params.batch_size)
  g_disable_dropout(model.rnns)
  for d = 1, 2 * model.params.layers do
    model.start_s[d]:zero()
  end
  g_replace_table(model.s[0], model.start_s)
  for i = 1, input:size(1) do
    local x = input[i]
    local y = x  -- or whatever
    _, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
    g_replace_table(model.s[0], model.s[1])
  end
  return model.s[1][4]
end

local function predict(model, text, n_best)
  -- Return :n_best: predictions from the :model: for the next word in :text:
  local out_state = get_output_state(model, text)
  local out_matrix = model.rnns[1]:parameters()[10] -- TODO - better way to address?
  -- FIXME - what is parameters()[11]???
  local pred_vec = out_state * out_matrix:transpose(1, 2)
  pred_vec = pred_vec[{1, {}}]
  local idx_prediction = {}
  for i = 1, pred_vec:size(1) do
    idx_prediction[i] = {i, pred_vec[i]}
  end
  table.sort(idx_prediction, function (a, b) return a[2] > b[2] end)
  local word_prediction = {}
  for i = 1, math.min(n_best, #idx_prediction) do
    x = idx_prediction[i]
    word_prediction[i] = {idx_to_word(model, x[1]), x[2]}
  end
  return word_prediction
end

return {
  predict=predict,
  get_output_state=get_output_state
}

;; WHY HY
;; - Python interop
;; - Lisp

(import itertools)
(import hyrule)
(import yaml)
(require hyrule [->])
(import json)


(defn car [l] (get l 0))
(defn cdr [l] (cut l 1 None))

(defn get* [mapping keys]
  (cond (= (len keys) 1) (get mapping (car keys))
        True (get* (get mapping (car keys)) (cdr keys))))




(defn var-to-tuple [var-sym] (tuple [(hy.models.Keyword var-sym) (hy.eval var-sym)]))
(defn symbols-to-value-config [var-syms] (dict (lfor vs var-syms (var-to-tuple vs))))

(defn lisp-to-python-name [varname] (.replace varname "-" "_"))
(defn grid-name-to-param-name [kw]
  (hy.models.Keyword (lisp-to-python-name (.replace (. kw name) "-grid" ""))))


;; this is like yaml but you can actually operate on configs
;;

(defn get-configs [config-param-grid]
  (setv names (lfor key (.keys config-param-grid) (grid-name-to-param-name key)))
  (print names)
  (lfor
    values (.product itertools (unpack-iterable (.values config-param-grid)))
    (dict (zip names values))))


(setv base-feature-names ["titles" "dependencies" "function_signature" "readme"]
      combined-feature-name "titles"
      query-embedder-grid ["nbow" "MiniLM" "simple_transformer"]
      document-embedder-grid ["nbow" "simple_transformer"]
      loss-function-name-grid ["multiple_negatives_ranking_loss"])
(print base-feature-names)
(setv feature-names-grid
      (+ base-feature-names (lfor ft base-feature-names :if (not (= ft "titles")) ["titles" ft])))

(defn get-model-param-grid []

  (symbols-to-value-config [`feature-names-grid `query-embedder-grid `document-embedder-grid `loss-function-name-grid]))


(setv global-env {
                  :metrics {:information_retrieval_metric "accuracy@10"}
                  :training {:epochs [1 20] :batch_size 128}
                  :logger-args {:neptune_config_path "neptune_stuff.json"}})
(setv param-grid (get-model-param-grid))

(setv model-configs (get-configs param-grid))

(setv model-configs-whole-dump ((. json dumps) model-configs))


(model-configs)

(defn get-tokenizer-config [feature]
  (dfor tpe ["document" "query"] tpe f"output/{feature}_{tpe}_tokenizer-0.pkl"))


(defn get-global-var [path])


;; 3 levels
;; env - stuff containing config and env-vars et c
;; config - dict that will actually get passed further
;; config-template - dict with expressions that will get evaluated using the types specified above

(setv training-config-template
      {
       :validation_metric_name `(get-global-var [:metrics :information_retrieval_metric])
       :epochs `(get-global-var [:training :epochs])
       :batch_size `(get-global-var [:training :batch_size])
       :model-config `(get-var model-env [:model-config])
       :logger-args `(get-global-var [:logger-args])
       :tokenizer_config '(get-tokenizer-config (get-main-feature model-env))
       :fasttext-path '(get-global-var [:fasttext-path])
       :data_args '(get-var data-env [:data-config])})

(defn get-main-feature [model-env]
  (get* model-env [:model-config :]))


(defn get-data-env []
  {:data-config
   {:data-path
    {:train "/home/kuba/Projects/github_search/output/nbow_data_train.parquet"
     :test "/home/kuba/Projects/github_search/output/nbow_data_test.parquet"}
    :paperswithcode-train-path "/home/kuba/Projects/github_search/output/repos_train.csv"}})

(defn get-model-env [] {:model-config (get model-configs 0)})

(defn get-env-expr [get-data-env get-model-env]
  `(setv model-env (get-model-env) data-env (get-data-env)))


(defn get-global-var [path]
  (get* global-env path))
(defn get-var [env var]
  (get* env var))


(defn config-template-expr-with-env [make-env-expr]
  (hy.eval make-env-expr)
  (hy.eval training-config-template))

(setv env-expr (get-env-expr get-data-env get-model-env))
(setv config (config-template-expr-with-env env-expr))

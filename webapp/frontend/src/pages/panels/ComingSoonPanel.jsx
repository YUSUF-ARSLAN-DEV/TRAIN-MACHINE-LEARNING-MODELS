export default function ComingSoonPanel({ model }) {
  return (
    <div className="panel coming-soon">
      <div className="coming-soon-icon">🚧</div>
      <h3>This model is still in the training queue</h3>
      <p>
        {model.shortName} is planned next on the roadmap ({model.task.replaceAll('-', ' ')}). Once
        it's fine-tuned and saved, this page will let you send it a live {model.inputType} input,
        the same way the BERT and ResNet-18 pages already work.
      </p>
    </div>
  )
}

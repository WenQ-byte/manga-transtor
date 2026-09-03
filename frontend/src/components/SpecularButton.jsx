import './SpecularButton.css'

const SpecularButton = ({
  children = 'Get Started',
  size = 'lg',
  radius = 12,
  variant = 'secondary',
  textColor = '#f5f5f5',
  icon = false,
  disabled = false,
  onClick,
  className = '',
  type = 'button',
  ...rest
}) => {

  return (
    <button
      type={type}
      disabled={disabled}
      onClick={onClick}
      className={`specular-button specular-button--${size} specular-button--${variant}${icon ? ' specular-button--icon' : ''}${className ? ` ${className}` : ''}`}
      style={{
        '--sb-radius': `${radius}px`,
        '--sb-text-color': textColor,
        borderRadius: `${radius}px`,
      }}
      {...rest}
    >
      <span className="specular-button__label">{children}</span>
    </button>
  );
};

export default SpecularButton;

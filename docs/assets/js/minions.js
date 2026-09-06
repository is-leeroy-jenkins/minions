document.addEventListener( 'DOMContentLoaded', ( ) => {
    document.querySelectorAll( '.highlight' ).forEach( ( block ) => {
        const code = block.querySelector( 'code' );
        if ( !code ) {
            return;
        }

        const languageClass = [ ...code.classList ].find( ( value ) =>
            value.startsWith( 'language-' )
        );
        if ( !languageClass ) {
            return;
        }

        const badge = document.createElement( 'span' );
        badge.className = 'minions-code-language';
        badge.textContent = languageClass.replace( 'language-', '' );
        block.appendChild( badge );
    } );
} );
